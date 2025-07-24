```
docker history myapp:latest
```
显示构建历史
```
➜  test git:(main) ✗ docker history myapp:latest
IMAGE          CREATED              CREATED BY                                      SIZE      COMMENT
ec63488b15fe   About a minute ago   /bin/sh -c #(nop)  ENTRYPOINT ["/app/app.sh"]   0B        
4e7d75d89c11   About a minute ago   /bin/sh -c #(nop) COPY file:ef4f5097986983d5…   0B        
145e4eda6aaf   About a minute ago   /bin/sh -c mkdir /app                           0B        
eac36dfb7e58   2 minutes ago        /bin/sh -c apt-get install -y curl              7.04MB    
2b50cafa0b51   2 minutes ago        /bin/sh -c apt-get update                       65.7MB    
1d3ca894b30c   9 days ago           /bin/sh -c #(nop)  CMD ["/bin/bash"]            0B        
<missing>      9 days ago           /bin/sh -c #(nop) ADD file:415bbc01dfb447d00…   77.9MB    
<missing>      9 days ago           /bin/sh -c #(nop)  LABEL org.opencontainers.…   0B        
<missing>      9 days ago           /bin/sh -c #(nop)  LABEL org.opencontainers.…   0B        
<missing>      9 days ago           /bin/sh -c #(nop)  ARG LAUNCHPAD_BUILD_ARCH     0B        
<missing>      9 days ago           /bin/sh -c #(nop)  ARG RELEASE                  0B      
```