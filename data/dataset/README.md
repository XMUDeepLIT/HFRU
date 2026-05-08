We conduct experiments on the PACS and VGGFace2 datasets.

Please download the [PACS](https://huggingface.co/datasets/flwrlabs/pacs) dataset yourself.

For the VGGFace2 dataset, we only use a subset of the data, so the filtered file (./vgg/vgg_select.parquet) is provided directly in this repository.

The dataset directory structure is as follows:

```text
dataset
├── pacs
│   └── data
│       └── train-00000-of-00001.parquet
└── vgg
    └── vgg_select.parquet
```

