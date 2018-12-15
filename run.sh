# bash get_gimages.sh

python keras_mem.py ./gimages-split --checkpoint_path gimages.h5 | tee results-gimages-split.txt
python keras_mem.py ./gimages-gray-split --checkpoint_path gimages-gray.h5 | tee results-gimages-gray-split.txt
python keras_mem.py ./NPYs_first10 --npy | tee results-quickdraw.txt
python keras_mem.py ./NPYs_first10 --npy --model_weights gimages.h5 | tee results-quickdraw-gimages.txt
python keras_mem.py ./NPYs_first10 --npy --model_weights gimages-gray.h5 | tee results-quickdraw-gimages-gray.txt

python keras_mem.py ./gimages_sketch-split --checkpoint_path sketch.h5 | tee results-sketch-split.txt
python keras_mem.py ./gimages_sketch-gray-split --checkpoint_path sketch-gray.h5 | tee results-sketch-gray-split.txt
python keras_mem.py ./NPYs_first10 --npy --model_weights sketch.h5 | tee results-quickdraw-sketch.txt
python keras_mem.py ./NPYs_first10 --npy --model_weights sketch-gray.h5 | tee results-quickdraw-sketch-gray.txt

python keras_mem.py ./gimages_simple-split --checkpoint_path simple.h5 | tee results-simple-split.txt
python keras_mem.py ./gimages_simple-gray-split --checkpoint_path simple-gray.h5 | tee results-simple-gray-split.txt
python keras_mem.py ./NPYs_first10 --npy --model_weights simple.h5 | tee results-quickdraw-simple.txt
python keras_mem.py ./NPYs_first10 --npy --model_weights simple-gray.h5 | tee results-quickdraw-simple-gray.txt
