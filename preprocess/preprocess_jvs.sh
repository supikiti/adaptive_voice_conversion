. preprocess/jvs.config

if [ $stage -le 0 ]; then
    python3.7 preprocess/make_datasets_jvs.py $raw_data_dir/ $data_dir $test_prop $n_utts_attr
fi

# if [ $stage -le 1 ]; then
#     python3 preprocess/reduce_dataset.py $data_dir/train.pkl $data_dir/train_$segment_size.pkl 
# fi

if [ $stage -le 2 ]; then
    # sample training samples
    python3 preprocess/sample_single_segments.py $data_dir/train.pkl $data_dir/train_samples_$segment_size.json $training_samples $segment_size
fi
if [ $stage -le 3 ]; then
    # sample testing samples
    python3.7 preprocess/sample_single_segments.py $data_dir/dev.pkl $data_dir/dev_samples_$segment_size.json $testing_samples $segment_size
    python3.7 preprocess/sample_single_segments.py $data_dir/test.pkl $data_dir/test_samples_$segment_size.json $testing_samples $segment_size
fi
