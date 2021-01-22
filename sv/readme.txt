train_place_embedding: train street view image embeddings on train pair, use val pair to filter
input: train_pair_ba.pickle, val_pair_ba.pickle
output: sv_embedding_.tar

embedding: generate embeddings for all street view images
input: image_ba.pickle
output: embedding_ba.tar, fips_ba.tar