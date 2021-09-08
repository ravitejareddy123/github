# A Docker Compose application with moqui, postgres, and virtual hosting through
# nginx-proxy supporting multiple moqui instances on different hosts.

# Run with something like this for detached mode:
# $ docker-compose -f moqui-ng-pg-compose.yml -p moqui up -d
# Or to copy runtime directories for mounted volumes, set default settings, etc use something like this:
# $ ./compose-run.sh moqui-ng-pg-compose.yml
# This sets the project/app name to 'moqui' and the network will be 'moqui_default', to be used by external moqui containers

# Test locally by adding the virtual host to /etc/hosts or with something like:
# $ curl -H "Host: moqui.local" localhost/Login

# To run an additional instance of moqui run something like this (but with
# many more arguments for volume mapping, db setup, etc):
# $ docker run -e VIRTUAL_HOST=moqui2.local --name moqui2_local --network moqui_default moqui

version: "2"
services:
  nginx-proxy:
    # For documentation on SSL and other settings see:
    # https://github.com/jwilder/nginx-proxy
    image: jwilder/nginx-proxy
    container_name: nginx-proxy
    restart: always
    ports:
      - 80:8000
      - 443:443
    volumes:
      - /var/run/docker.sock:/tmp/docker.sock:ro
      # note: .crt, .key, and .dhparam.pem files start with the domain name in VIRTUAL_HOST (ie 'moqui.local.*') or use CERT_NAME env var
      - ./certs:/etc/nginx/certs
      - ./nginx/my_proxy.conf:/etc/nginx/conf.d/my_proxy.conf
    environment:
      # change this for the default host to use when accessing directly by IP, etc
      - DEFAULT_HOST=moqui.local
      # use SSL_POLICY to disable TLSv1.0, etc in nginx-proxy
      - SSL_POLICY=AWS-TLS-1-1-2017-01

  moqui-server:
    image: raviteja1996/ravi
    container_name: moqui-server
    command: conf=conf/MoquiProductionConf.xml
    restart: always
    links:
     - moqui-database
     - elasticsearch
    volumes:
     - ./runtime/conf:/opt/moqui/runtime/conf
     - ./runtime/lib:/opt/moqui/runtime/lib
     - ./runtime/classes:/opt/moqui/runtime/classes
     - ./runtime/ecomponent:/opt/moqui/runtime/ecomponent
     - ./runtime/log:/opt/moqui/runtime/log
     - ./runtime/txlog:/opt/moqui/runtime/txlog
     - ./runtime/sessions:/opt/moqui/runtime/sessions
     # this one isn't needed when not using H2/etc: - ./runtime/db:/opt/moqui/runtime/db
    environment:
     - instance_purpose=production
     - entity_ds_db_conf=postgres
     - entity_ds_host=moqui-database
     - entity_ds_port=5432
     - entity_ds_database=moqui
     - entity_ds_schema=public
     - entity_ds_user=moqui
     - entity_ds_password=moqui
     - entity_ds_crypt_pass='MoquiDefaultPassword:CHANGEME'
     # configuration for ElasticFacade.ElasticClient, make sure moqui-elasticsearch is NOT included in the Moqui build
     - elasticsearch_url=http://elasticsearch:9200
     # CHANGE ME - note that VIRTUAL_HOST is for nginx-proxy so it picks up this container as one it should reverse proxy
     # this can be a comma separate list of hosts like 'example.com,www.example.com'
     - VIRTUAL_HOST=moqui.local
     # moqui will accept traffic from other hosts but these are the values used for URL writing when specified:
     - webapp_http_host=moqui.local
     - webapp_http_port=80
     - webapp_https_port=443
     - webapp_https_enabled=true
     - default_locale=en_US
     - default_time_zone=US/Pacific

  moqui-database:
    image: postgres:12.1
    container_name: moqui-database
    restart: always
    ports:
     # change this as needed to bind to any address or even comment to not expose port outside containers
     - 127.0.0.1:5432:5432
    volumes:
     # edit these as needed to map configuration and data storage
     - ./db/postgres/data:/var/lib/postgresql/data
    environment:
     - POSTGRES_DB=moqui
     - POSTGRES_DB_SCHEMA=public
     - POSTGRES_USER=moqui
     - POSTGRES_PASSWORD=moqui
     # PGDATA, POSTGRES_INITDB_ARGS

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch-oss:7.4.2
    container_name: elasticsearch
    restart: always
    ports:
      # change this as needed to bind to any address or even comment to not expose port outside containers
      - 127.0.0.1:9200:9200
      - 127.0.0.1:9300:9300
    volumes:
      # edit these as needed to map configuration and data storage
      - ./elasticsearch/data:/usr/share/elasticsearch/data
      # - ./elasticsearch/config/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml
      # - ./elasticsearch/logs:/usr/share/elasticsearch/logs
    environment:
      - "ES_JAVA_OPTS=-Xms1024m -Xmx1024m"
      - discovery.type=single-node
      - network.host=_site_
