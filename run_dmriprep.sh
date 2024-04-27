#!/bin/bash
if [[ $1==local ]]; then
    ROOT='/Users/himanshu/Desktop/diffusion'
else
    ROOT='/storage/store3/work/haggarwa/diffusion'
fi
BIDS_DATA=$ROOT'/bids_data'
OUTPUT_DIR=$ROOT'/result'
docker run -it --rm -v $BIDS_DATA:/data:ro -v $OUTPUT_DIR:/out $2 /data /out/out participant --verbose --fs-no-reconall