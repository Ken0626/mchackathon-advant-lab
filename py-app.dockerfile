


FROM unifiedserver.local/all/template-data-app:v22.04
WORKDIR /opt/nexus/OneAPI
COPY bin/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY bin/. ./bin
ENV ONEAPI_DEBUG 6
WORKDIR /opt/nexus/OneAPI/bin
CMD ["python3", "-u", "main.py"]


