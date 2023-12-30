Reformat of Bayesian Invariant Risk Minimization

To run the first code: run with

```bash
python train.py --l2_regularizer_weight 0.004 --hidden_dim 3 --_lambda_ 100 --steps 1000 --lr 0.0004
```

To run the second code: run with
```bash
python main.py --l2_regularizer_weight 0.004 --lr 0.0004 --print_every -1 --hidden_dim 1000 --penalty_weight 60000 --steps 600 --data_num 20000 --seed 0 --wandb_log_freq -1 --model MLP --shape 28 --sampleN 100 --device 0
```

The first code does not need gpu since I tried on colab and found it not improved much. 