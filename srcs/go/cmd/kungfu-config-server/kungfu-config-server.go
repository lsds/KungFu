package main

import (
	"context"
	"flag"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/elastic/configserver"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	host     = flag.String("host", "0.0.0.0", "")
	port     = flag.Int("port", 9100, "")
	initFile = flag.String("init", "", "")
	ttl      = flag.Duration("ttl", 0, "time to live")
	endpoint = flag.String("endpoint", "/config", "URL path for Rest API")
)

func main() {
	t0 := time.Now()
	flag.Parse()
	listenURL := url.URL{
		Scheme: `http`,
		Host:   net.JoinHostPort(*host, strconv.Itoa(*port)),
		Path:   *endpoint,
	}
	log.Infof("listening %s", listenURL.String())
	var initCluster *plan.Cluster
	if len(*initFile) > 0 {
		f, err := os.Open(*initFile)
		if err != nil {
			utils.ExitErr(err)
		}
		defer f.Close()
		if err := utils.ReadJSON(f, &initCluster); err != nil {
			utils.ExitErr(err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	if *ttl > 0 {
		ctx, cancel = context.WithTimeout(ctx, *ttl)
		defer cancel()
	}
	srv := &http.Server{
		Addr:    net.JoinHostPort("", strconv.Itoa(*port)),
		Handler: logRequest(configserver.New(cancel, initCluster, *endpoint)),
	}
	srv.SetKeepAlivesEnabled(false)
	defer srv.Close()
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			utils.ExitErr(err)
		}
	}()
	<-ctx.Done()
	log.Infof("%s stopped after %s", utils.ProgName(), time.Since(t0))
}

func logRequest(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		log.Debugf("%s %s from %s, UA: %s", req.Method, req.URL.Path, req.RemoteAddr, req.UserAgent())
		h.ServeHTTP(w, req)
	})
}
