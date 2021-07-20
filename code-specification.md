# Long-Tail Hashing

## REQUIREMENTS
`pip install -r requirements.txt`

1. pytorch
2. loguru

## DATASETS
1. [Cifar-100]
2. [ImageNet-100]

## USAGE
```
usage: python run.py [-h] [--dataset DATASET] [--root ROOT] [--batch-size BATCH_SIZE]
              [--arch ARCH] [--lr LR] [--code-length CODE_LENGTH]
			  [--feature-dim FEATURE_DIM] [--num-classes NUM_CLASSES] [--num-prototypes NUM_PROTOTYPES]
			  [--dynamic-meta-embedding DYNAMIC_META_EMBEDDING]
              [--max-iter MAX_ITER]
              [--num-train NUM_TRAIN] [--num-workers NUM_WORKERS]
              [--topk TOPK] [--gpu GPU] [--beta BETA]
              [--evaluate-interval EVALUATE_INTERVAL]

LTH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --batch-size          BATCH_SIZE
                        Batch size.(default: 16)
  --arch ARCH           CNN model name.(default: resnet34)
  --lr LR               Learning rate.(default: 1e-5)
  --code-length CODE_LENGTH
                        Binary hash code length.(default: 32,48,64,96)
  --max-iter MAX_ITER   Number of iterations.(default: 100)
  --feature-dim FEATURE_DIM
                        Dimension of feature. (default: 2000)
  --num-classes NUM_CLASSES
                        Number of classes. (default: 100)
  --num-prototypes NUM_PROTOTYPES
                        Number of prototypes. (default: 100)
  --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 6)
  --topk TOPK           Calculate map of top k.(default: all)
  --gpu GPU             Using gpu.(default: False)
  --mu MU               Hyper-parameter.(default: 1e-2)
  --nu NU               Hyper-parameter.(default: 1)
  --eta ETA             Hyper-parameter.(default: 1e-2)
  --evaluate-interval EVALUATE_INTERVAL
                        Evaluation interval.(default: 1)
```

## EXPERIMENTS
Model: resnet34 + dynamic meta embedding. Compute mean average precision (MAP).

Cifar100: 100 classes, query images, training images, database images.

ImageNet-100: 100 classes, query images, training images, database images.

