if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/SegRNN" ]; then
    mkdir ./logs/SegRNN
fi
model_name=SegRNN

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

seq_len=720
for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --patch_len 48 \
      --stride 48 \
      --enc_in 7 \
      --d_model 512 \
      --dropout 0.5 \
      --train_epochs 30 \
      --patience 10 \
      --itr 1 --batch_size 256 --learning_rate 0.001
done

