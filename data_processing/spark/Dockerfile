FROM bitnami/spark:3.4.1

# Set working directory
WORKDIR /opt/bitnami/spark

# Copy Spark jobs into the container
COPY jobs /opt/bitnami/spark/jobs

# Expose Spark master and worker ports
EXPOSE 7077 8080

# Default command to run Spark
CMD ["/opt/bitnami/scripts/spark/entrypoint.sh"]
