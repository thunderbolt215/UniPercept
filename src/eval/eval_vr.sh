# Example for single device
python src/eval/eval_vr.py  \
        --model_name unipercept \
        --datasets ArtiMuse-10K,KonIQ-10K,ISTA-10K \
        --devices 0

# Example for multi devices
python src/eval/eval_vr.py  \
        --model_name unipercept \
        --datasets ArtiMuse-10K,KonIQ-10K,ISTA-10K,AVA,TAD66K,FLICKR-AES,SPAQ,KADID,PIPAL \
        --devices 0,1,2,3,4,5,6,7