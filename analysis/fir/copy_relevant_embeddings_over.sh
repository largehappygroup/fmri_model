#!/bin/bash

  REMOTE="zachkaras@cumberland.isis.vanderbilt.edu:/s3/zach/fmri_model_data/fir_vectors_pca_params/"
  LOCAL="/data/zachkaras/fmri_model_data/fir_vectors_pca_params/"

  rsync -av --progress \
      --include="*/" \
      --include="codegemma_7b-code-look_ahead_by_10-ndelays_4-fir_embedding-layer_4*" \
      --include="codegemma_7b-code-look_ahead_by_10-ndelays_4-fir_embedding-layer_16*" \
      --include="codegemma_7b-prose-look_ahead_by_5-ndelays_10-fri_embedding-layer_4*" \
      --include="codegemma_7b-prose-look_ahead_by_5-ndelays_10-fri_embedding-layer_16*" \
      --exclude="*" \
      "$REMOTE" "$LOCAL"