#!/bin/sh
set -e

GROUP=kungfu
LOCATION=southeastasia

az resource list -l ${LOCATION} -g ${GROUP} -o table | sed 1,2d |
    awk '{print "az resource delete -g kungfu --resource-type ", $4, "-n ", $1, "--debug"}'
