The code will be updated soon.



#### Train

##### IEMOCAP

```bash
python main.py -backbone ./pretrained_model -run_type train -dataset iemocap -use_gat -window_size 8 -emotion_first -use_video_mode -use_audio_mode
```

##### MELD

```bash
python main.py -backbone ./pretrained_model -run_type train -dataset meld -use_gat -emotion_first -use_video_mode -use_audio_mode
```

#### Predict

##### IEMOCAP

```bash
python main.py -run_type predict -ckpt ./iemocap-best-model/ckpt -output predict_real.json -dataset iemocap -test_batch_size=64
```

##### MELD

```bash
python main.py -run_type predict -ckpt ./meld-best-model/ckpt -output predict_real.json -dataset meld -test_batch_size=64
```

