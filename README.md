# VOLTA

PyTorch code for VOLTA (TIP 2020) described in the paper "Iterative Local-Global Collaboration Learning towards One-Shot Video Person Re-Identification" [[Link]](https://ieeexplore.ieee.org/document/9211791). This code is based on the [Open-ReID](https://github.com/Cysu/open-reid) library and [EUG](<https://github.com/Yu-Wu/Exploit-Unknown-Gradually>). 

## Preparation
### Dependencies
- Python 3.5
- PyTorch (version >= 1.0.1)
- h5py, scikit-learn, metric-learn, tqdm

### Download datasets 
- DukeMTMC-VideoReID: This [page](https://github.com/Yu-Wu/DukeMTMC-VideoReID) contains more details and baseline code.
- MARS: [[Google Drive]](https://drive.google.com/open?id=1m6yLgtQdhb6pLCcb6_m7sj0LLBRvkDW0)   [[BaiduYun]](https://pan.baidu.com/s/1mByTdvXFsmobXOXBEkIWFw).
- Move the downloaded zip files to `./data/` and unzip here.


## Train

For the DukeMTMC-VideoReID dataset:
```shell
python3 run.py --dataset DukeMTMC-VideoReID --logs_dir logs/run_duke/ --max_frames 900 --init_lr 0.1
```

For the MARS datasaet:
```shell
python3 run.py --dataset mars --logs_dir logs/run_mars/ --max_frames 900 --init_lr 0.05
```
Please set the `max_frames` smaller if your GPU memory is not enough. 

## Performances

The performances varies according to random splits for initial labeled data. For fair comparisons, we follow the same one-shot splits used in [EUG](<https://github.com/Yu-Wu/Exploit-Unknown-Gradually>). To reproduce the performances in our paper, please use the one-shot splits at `./examples/`.


## Citation

Please cite the following paper in your publications if it helps your research: 

```
@article{liu2020iterative,
  title  = {Iterative Local-Global Collaboration Learning towards One-Shot Video Person Re-Identification},
  author = {Liu, Meng and Qu, Leigang and Nie, Liqiang and Liu, Maofu, and Duan, Lingyu and Chen, Baoquan},
  journal= {IEEE Transactions on Image Processing},
  year   = {2020}, 
  volume = {}, 
  number = {}, 
  pages  = {}, 
  doi    = {}, 
  ISSN   = {}, 
  month  = {},
} 
```

