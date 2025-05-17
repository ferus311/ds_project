from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import joblib
import pandas as pd
import json
import requests
import numpy as np

# --- MODEL & PREPROCESSOR LOADING ---
# Lý tưởng nhất, các thành phần này nên được tải một lần khi ứng dụng Django khởi động.
# Ví dụ: trong apps.py của app prediction_api hoặc một module riêng và import vào đây.
# Để đơn giản cho ví dụ này, chúng ta sẽ cố gắng tải chúng bên trong view,
# nhưng có kèm theo cơ chế cache đơn giản để tránh tải lại mỗi request.
MODEL_COMPONENTS_CACHE = {}

def load_model_and_preprocessors():
    if not MODEL_COMPONENTS_CACHE: # Chỉ tải nếu cache rỗng
        try:
            model_data_dict = joblib.load(settings.MODEL_FILE_PATH)
            MODEL_COMPONENTS_CACHE['model'] = model_data_dict['model']
            MODEL_COMPONENTS_CACHE['scaler'] = model_data_dict['scaler']
            MODEL_COMPONENTS_CACHE['numerical_cols'] = model_data_dict['numerical_cols']
            MODEL_COMPONENTS_CACHE['clip_dict'] = model_data_dict['clip_dict']
            print(f"Model components loaded successfully from {settings.MODEL_FILE_PATH}")
            # print(f"DEBUG: Numerical cols for scaling: {MODEL_COMPONENTS_CACHE['numerical_cols']}")
            # print(f"DEBUG: Clip dict: {MODEL_COMPONENTS_CACHE['clip_dict']}")
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Model file not found at {settings.MODEL_FILE_PATH}")
            # Để trống cache, các request sau sẽ báo lỗi
        except KeyError as e:
            print(f"CRITICAL ERROR: Expected key {e} not found in the loaded model file from {settings.MODEL_FILE_PATH}")
        except Exception as e:
            print(f"CRITICAL ERROR: Error loading model components from {settings.MODEL_FILE_PATH}: {e}")
    return MODEL_COMPONENTS_CACHE

JSON_INPUT_COLUMNS = [
    'bedroom', 'bathroom', 'facade_width', 'road_width', 'floor_count',
    'price_vnd',
    'price_per_m2_vnd',
    'area_m2', 'latitude', 'longitude', 'price_per_m2_ratio', 'area_ratio',
    'area_per_room', 'comfort_index', 'facade_area_ratio', 'road_facade_ratio',
    'distance_to_center', 'location_cluster'
]

COLS_FILL_MINUS_ONE_IF_MISSING = [
    'bedroom', 'bathroom', 'facade_width', 'road_width', 'floor_count',
    'facade_area_ratio', 'road_facade_ratio', 'area_per_room'
]


COLS_REQUIRE_MEDIAN_IF_MISSING_AND_NOT_GEOCDDED = ['price_per_m2_ratio']


# --- Hàm Helper cho Geocoding (giữ nguyên từ trước) ---
def _get_coordinates_from_address(address_text):
    api_key = getattr(settings, 'GOOGLE_GEOCODING_API_KEY', None)
    if not api_key:
        raise ValueError("Google Geocoding API key not configured.")
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address_text}&key={api_key}&region=VN&language=vi"
    try:
        response = requests.get(geocode_url, timeout=10)
        response.raise_for_status()
        response_data = response.json()
    except requests.exceptions.Timeout:
        raise ConnectionError("Geocoding request timed out.")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Network error calling Google Geocoding API: {str(e)}")

    if response_data.get('status') == 'OK' and response_data.get('results'):
        location = response_data['results'][0]['geometry']['location']
        return location['lat'], location['lng'], response_data['results'][0].get('formatted_address', address_text)
    else:
        error_message = response_data.get('error_message', 'Unknown error from Google Geocoding API.')
        google_status = response_data.get('status', 'UNKNOWN_STATUS')
        if google_status == 'ZERO_RESULTS':
            raise ValueError(f"No results found for the address: {address_text}. Google status: {google_status}")
        else:
            raise ConnectionError(f"Error from Google Geocoding API: {error_message}. Status: {google_status}")

@csrf_exempt
def geocode_address_api(request):
    if request.method == 'POST':
        try:
            if request.content_type == 'application/json':
                data = json.loads(request.body)
            else:
                return JsonResponse({'error': 'Invalid content type. Please send JSON data.'}, status=400)
            address_text = data.get('address')
            if not address_text:
                return JsonResponse({'error': 'Missing address field in JSON body.'}, status=400)
            latitude, longitude, formatted_address = _get_coordinates_from_address(address_text)
            return JsonResponse({
                'requested_address': address_text,
                'formatted_address': formatted_address,
                'latitude': latitude,
                'longitude': longitude
            })
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data provided.'}, status=400)
        except ValueError as ve:
            return JsonResponse({'error': str(ve)}, status=400 if "No results" in str(ve) else 500)
        except ConnectionError as ce:
            return JsonResponse({'error': str(ce)}, status=503)
        except Exception as e:
            return JsonResponse({'error': f'Unexpected error during geocoding: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


@csrf_exempt
def predict_house_price_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

    components = load_model_and_preprocessors()
    if not components or 'model' not in components or 'scaler' not in components or \
       'numerical_cols' not in components or 'clip_dict' not in components:
        return JsonResponse({'error': 'Model or critical preprocessor components not loaded. Check server logs.'}, status=500)

    model = components['model']
    scaler = components['scaler']
    numerical_cols_from_training = components['numerical_cols'] # List of column names for scaling
    clip_dict = components['clip_dict']               # Dict {'col_name': upper_bound}

    try:
        if request.content_type == 'application/json':
            raw_data_from_json = json.loads(request.body)
        else:
            return JsonResponse({'error': 'Invalid content type. Please send JSON data.'}, status=400)

        # --- 1. Chuẩn bị dữ liệu đầu vào (bao gồm geocoding nếu cần) ---
        processed_input_values = {}
        final_latitude = None
        final_longitude = None

        input_latitude = raw_data_from_json.get('latitude')
        input_longitude = raw_data_from_json.get('longitude')
        address_text = raw_data_from_json.get('address')

        if input_latitude is not None and input_longitude is not None:
            try:
                final_latitude = float(input_latitude)
                final_longitude = float(input_longitude)
            except ValueError:
                return JsonResponse({'error': 'Invalid latitude/longitude provided. Must be numeric.'}, status=400)
        elif address_text:
            try:
                lat, lon, _ = _get_coordinates_from_address(address_text)
                final_latitude = lat
                final_longitude = lon
            except (ValueError, ConnectionError) as geo_err:
                return JsonResponse({'error': f'Geocoding failed: {str(geo_err)}'}, status=400 if isinstance(geo_err, ValueError) else 503)
            except Exception as e:
                return JsonResponse({'error': f'Geocoding failed with unexpected error: {str(e)}'}, status=500)
        
        # Lấy các giá trị feature từ JSON hoặc đã geocode, xử lý missing cơ bản
        for feature_name in JSON_INPUT_COLUMNS:
            if feature_name == 'latitude':
                value_to_process = final_latitude
            elif feature_name == 'longitude':
                value_to_process = final_longitude
            else:
                value_to_process = raw_data_from_json.get(feature_name)

            if value_to_process is None or (isinstance(value_to_process, str) and str(value_to_process).lower() == 'nan'):
                if feature_name in COLS_FILL_MINUS_ONE_IF_MISSING:
                    processed_input_values[feature_name] = -1.0
                elif feature_name in ['latitude', 'longitude']: # Đã xử lý ở trên, nếu vẫn None là lỗi
                     if final_latitude is None or final_longitude is None: # Check lại phòng trường hợp logic thay đổi
                        return JsonResponse({'error': f'Missing location data: Provide valid lat/lon or address for {feature_name}.'}, status=400)
                elif feature_name in COLS_REQUIRE_MEDIAN_IF_MISSING_AND_NOT_GEOCDDED:
                     # Chỗ này cần lấy median_value từ settings hoặc 1 config nào đó
                     # Ví dụ: median_val = settings.MEDIAN_VALUES_FOR_IMPUTATION.get(feature_name)
                     # if median_val is not None: processed_input_values[feature_name] = float(median_val) else: ... lỗi ...
                    return JsonResponse({'error': f'Missing feature: {feature_name}, and median value not configured for imputation.'}, status=400)
                else: # Các feature khác nếu thiếu sẽ báo lỗi (trừ khi nó được tạo ra sau)
                    # price_vnd, price_per_m2_vnd, area_m2 là critical inputs cho feature engineering
                    if feature_name in ['price_vnd', 'price_per_m2_vnd', 'area_m2']:
                         return JsonResponse({'error': f'Missing critical feature for processing: {feature_name}'}, status=400)
                    # Các feature còn lại nếu thiếu và không nằm trong list fill -1, thì cũng lỗi
                    # Hoặc chúng sẽ được xử lý/tạo ra sau (price_per_area, room_density)
                    # Nếu không, giá trị mặc định là None sẽ được đưa vào DataFrame, pandas có thể xử lý sau
                    # nhưng code notebook có vẻ không mong đợi điều này cho các feature input cơ bản.
                    processed_input_values[feature_name] = None # Sẽ được kiểm tra sau nếu cần thiết

            else: # Value is present
                try:
                    if feature_name not in ['latitude', 'longitude']: # Lat/Lon đã là float
                         processed_input_values[feature_name] = float(value_to_process)
                    else:
                         processed_input_values[feature_name] = value_to_process # giữ nguyên final_latitude/longitude đã là float
                except ValueError:
                    return JsonResponse({'error': f'Invalid value for feature {feature_name}: {value_to_process}. Must be numeric.'}, status=400)
        
        # Tạo DataFrame từ dữ liệu đã xử lý sơ bộ
        df = pd.DataFrame([processed_input_values])

        # --- 2. Áp dụng các phép biến đổi như trong notebook ---

        # 2a. Xử lý missing value (điền -1) - Đã làm ở bước 1 cho processed_input_values

        # 2b. Xử lý missing value cho các cột còn lại (điền median) - Đã giải quyết ở bước 1, yêu cầu config median

        # 2c. Tính lại area_per_room (nếu có thể) - Logic từ notebook
        # Đảm bảo các cột cần thiết có mặt và là số
        required_cols_ar = ['area_per_room', 'bedroom', 'bathroom', 'area_m2']
        if all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in required_cols_ar):
            mask = (df['area_per_room'] == -1) & (df['bedroom'] > 0) & (df['bathroom'] > 0) & (df['area_m2'] > 0) # Thêm area_m2 > 0
            if mask.any():
                room_sum = df.loc[mask, 'bedroom'] + df.loc[mask, 'bathroom']
                # Tránh chia cho 0 nếu room_sum = 0 (mặc dù mask có bedroom > 0, bathroom > 0)
                df.loc[mask, 'area_per_room'] = df.loc[mask, 'area_m2'] / room_sum.replace(0, np.nan) # Thay 0 bằng NaN để tránh lỗi, fillna sau nếu cần
                df['area_per_room'].fillna(-1, inplace=True) # Nếu chia cho NaN thì fill lại -1

        # 2d. Cắt ngưỡng outlier (sử dụng clip_dict từ model)
        for col_name, upper_bound in clip_dict.items():
            if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                # Chỉ clip nếu giá trị > upper_bound VÀ không phải là giá trị đã điền (-1)
                df.loc[(df[col_name] > upper_bound) & (df[col_name] != -1.0), col_name] = float(upper_bound)

        # 2e. Tạo đặc trưng mới: price_per_area, room_density
        # Cần đảm bảo các cột input là số và hợp lệ (vd: area_m2 > 0)
        if 'price_vnd' in df.columns and 'area_m2' in df.columns and \
            pd.api.types.is_numeric_dtype(df['price_vnd']) and pd.api.types.is_numeric_dtype(df['area_m2']) and df['area_m2'].iloc[0] > 0:
            df['price_per_area'] = df['price_vnd'] / df['area_m2']
        else:
            df['price_per_area'] = 0.0 # Giá trị mặc định hoặc xử lý lỗi khác

        if 'bedroom' in df.columns and 'bathroom' in df.columns and 'area_m2' in df.columns and \
            pd.api.types.is_numeric_dtype(df['bedroom']) and pd.api.types.is_numeric_dtype(df['bathroom']) and \
            pd.api.types.is_numeric_dtype(df['area_m2']) and df['area_m2'].iloc[0] > 0:
            df['room_density'] = (df['bedroom'] + df['bathroom']) / df['area_m2']
        else:
            df['room_density'] = 0.0 # Giá trị mặc định

        # 2f. Log-transform: price_per_m2_vnd
        if 'price_per_m2_vnd' in df.columns and pd.api.types.is_numeric_dtype(df['price_per_m2_vnd']):
            df['price_per_m2_vnd'] = np.log1p(df['price_per_m2_vnd'])
        else: # Nếu price_per_m2_vnd không có hoặc không phải số, thì đây là lỗi thiếu input nghiêm trọng
            return JsonResponse({'error': 'Missing or invalid price_per_m2_vnd for log transformation.'}, status=400)

        # --- 3. Chuẩn hóa dữ liệu (Scaling) ---
        # Chỉ chuẩn hóa các cột số có trong numerical_cols_from_training và có mặt trong df
        cols_to_scale = [col for col in numerical_cols_from_training if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if cols_to_scale:
            try:
                df[cols_to_scale] = scaler.transform(df[cols_to_scale])
            except Exception as e:
                 # Có thể lỗi do scaler không fit với số cột, hoặc kiểu dữ liệu
                print(f"Error during scaling: {e}. Columns to scale: {cols_to_scale}. Dtypes: {df[cols_to_scale].dtypes}")
                return JsonResponse({'error': f'Error during data scaling: {e}'}, status=500)
        
        # --- 4. Chọn features và thứ tự cho model ---
        # Dựa trên notebook: X = df[[col for col in df.columns if col != 'price_vnd']]
        # Thứ tự các cột trong df tại thời điểm này sẽ là thứ tự cho model.
        # 'price_vnd' đã được sử dụng để tạo 'price_per_area' và không đưa vào model.
        
        # Tạo danh sách các cột cuối cùng cho model dựa trên các cột hiện có trong df, trừ 'price_vnd'
        # Thứ tự được xác định bởi thứ tự các cột trong df tại thời điểm này.
        # Notebook không định nghĩa một list tường minh, mà lấy tất cả trừ 'price_vnd'.
        # Để an toàn, chúng ta nên định nghĩa một list tường minh các feature model mong đợi.
        # Đây là những features có trong df SAU TẤT CẢ BIẾN ĐỔI và TRỪ 'price_vnd'
        MODEL_FEATURES_ORDER = [
            'bedroom', 'bathroom', 'facade_width', 'road_width', 'floor_count',
            'price_per_m2_vnd', # Đã log-transform
            'area_m2', 'latitude', 'longitude', 'price_per_m2_ratio', 'area_ratio',
            'area_per_room', 'comfort_index', 'facade_area_ratio', 'road_facade_ratio',
            'distance_to_center', 'location_cluster',
            'price_per_area',   # Feature mới
            'room_density'      # Feature mới
        ]
        
        # Kiểm tra xem tất cả các feature model cần có trong df không
        missing_model_features = [mf for mf in MODEL_FEATURES_ORDER if mf not in df.columns]
        if missing_model_features:
            return JsonResponse({'error': f'Internal error: Following features required by model are missing after processing: {missing_model_features}'}, status=500)

        final_input_df = df[MODEL_FEATURES_ORDER]
        
        # --- 5. Dự đoán và chuyển đổi ngược ---
        prediction_log = model.predict(final_input_df)
        predicted_price = np.expm1(prediction_log[0])

        return JsonResponse({'predicted_price': round(predicted_price, 0)}) # Làm tròn

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data provided.'}, status=400)
    except KeyError as e: # Bắt lỗi nếu truy cập key không tồn tại trong processed_input_values hoặc df
        return JsonResponse({'error': f'Missing expected data field during processing: {str(e)}'}, status=400)
    except Exception as e:
        print(f"Unexpected error during prediction: {type(e).__name__} - {str(e)}") # Log lỗi chi tiết hơn
        import traceback
        traceback.print_exc() # In traceback cho debug trên server
        return JsonResponse({'error': f'An unexpected error occurred: {str(e)}'}, status=500) 