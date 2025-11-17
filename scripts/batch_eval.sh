# RUNNING INFERENCE
# predetermined target camera view mode
python inference/inference.py --view \
  --model_ckpt weight/larm/model_299000.pth \
  --datalist_path data_sample/view_metadata/data.txt \
  --save_dir output_view \
  --resolution 512 --batch_size 4 --num_input_views 6 \
  --qpos_in_a 0.00 --qpos_in_b 1.00

# random mode
python inference/inference.py --random \
  --model_ckpt weight/larm/model_299000.pth \
  --datalist_path data_sample/random_metadata/data.txt \
  --save_dir output_random \
  --resolution 512 --batch_size 4 --num_input_views 6 \
  --qpos_in_a 0.00 --qpos_in_b 1.00 --num_target_views=64

# # joint estimation for urdf
cd axis_est
python estimate_ransac.py --load_dir=../output_random --datalist_path=../data_sample/view_metadata/data.txt

# # reconstruction with SAP (set up the environment as instructed in the mesh folder README)
python mesh/SAP/preprocess_sap.py
cd mesh/SAP/shape_as_points
python generate.py sap.yaml
cd ../..
python mesh/SAP/postprocess_sap.py

# # run metrics script, can access different modes:
# joint parameters mode
python metrics/eval.py --joint \
  --eval_list ./data_sample/view_metadata/data.txt \
  --category_json ./data_sample/train_test_split.json \
  --urdf_root ./output_view \
  --joint_info_json ./data_sample/joint_info.json \
  --categories ALL