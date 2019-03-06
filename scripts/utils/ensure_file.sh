ensure_file() {
    local sha1=$1
    local filename=$2
    local URL=$3

    if [ -f $filename ]; then
        if [ $(sha1sum $filename | awk '{print $1}') != "$sha1" ]; then
            echo "$filename has invalid sha1 sum"
            rm $filename
        else
            echo "use existing file $filename"
            return
        fi
    fi

    curl -vLOJ $URL
    if [ ! -f $filename ]; then
        echo "Download $filanem failed"
        return 1
    fi
    if [ $(sha1sum $filename | awk '{print $1}') != $sha1 ]; then
        echo "downloaded file has invalid sha1"
        return 1
    fi
}
