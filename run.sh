# bash get_gimages.sh

python keras_mem.py ./NPYs_first10 --no_augmentation --checkpoint_path quickdraw.h5 --npy | tee results-quickdraw.txt
# python keras_mem.py ./gimages-split --checkpoint_path gimages.h5 | tee results-gimages-split.txt
# python keras_mem.py ./gimages-gray-split --checkpoint_path gimages-gray.h5 | tee results-gimages-gray-split.txt

# python keras_mem.py ./NPYs_first10 --no_augmentation --model_weights gimages.h5 --checkpoint_path quickdraw-gimages.h5  --npy | tee results-quickdraw-gimages.txt
# python keras_mem.py ./NPYs_first10 --no_augmentation --model_weights gimages-gray.h5 --checkpoint_path quickdraw-gimages-gray.h5 --npy | tee results-quickdraw-gimages-gray.txt

# python keras_mem.py ./gimages_sketch-split --checkpoint_path sketch.h5 | tee results-sketch-split.txt
# python keras_mem.py ./gimages_sketch-gray-split --checkpoint_path sketch-gray.h5 | tee results-sketch-gray-split.txt
# python keras_mem.py ./NPYs_first10 --no_augmentation --model_weights sketch.h5 --checkpoint_path quickdraw-sketch.h5 --npy | tee results-quickdraw-sketch.txt
# python keras_mem.py ./NPYs_first10 --no_augmentation --model_weights sketch-gray.h5 --checkpoint_path quickdraw-sketch-gray.h5 --npy | tee results-quickdraw-sketch-gray.txt

# python keras_mem.py ./gimages_simple-split --checkpoint_path simple.h5 | tee results-simple-split.txt
# python keras_mem.py ./gimages_simple-gray-split --checkpoint_path simple-gray.h5 | tee results-simple-gray-split.txt
# python keras_mem.py ./NPYs_first10 --no_augmentation --model_weights simple.h5 --checkpoint_path quickdraw-simple.h5 --npy | tee results-quickdraw-simple.txt
# python keras_mem.py ./NPYs_first10 --no_augmentation --model_weights simple-gray.h5 --checkpoint_path quickdraw-simple-gray.h5 --npy | tee results-quickdraw-simple-gray.txt

# python keras_mem.py cifar10 --no_augmentation --checkpoint_path cifar10.h5 | tee results-cifar10.txt
# python keras_mem.py ./NPYs_first10 --no_augmentation --model_weights cifar10.h5 --checkpoint_path quickdraw-cifar10.h5 --npy | tee results-quickdraw-after-cifat10.txt
python keras_mem.py cifar10 --no_augmentation --model_weights quickdraw.h5 --checkpoint_path cifar10-quickdraw.h5 --npy | tee results-cifar10-after-quickdraw.txt
# python keras_mem.py cifar10 --no_augmentation --model_weights gimages.h5 --checkpoint_path gimages-cifar10.h5 | tee results-cifar10-after-gimages.txt
# python keras_mem.py ./gimages-split  --model_weights cifar10.h5 --checkpoint_path gimages-cifar10.h5 | tee results-gimages-after-cifar10.txt

# test only
# python keras_mem.py cifar10 --no_augmentation --checkpoint_path cifar10.h5 | tee results-cifar10.txt
# python keras_mem.py ./NPYs_first10 --no_augmentation --model_weights quickdraw-cifar10.h5 --test_only --npy| tee test-results-quickdraw-after-cifat10.txt
# python keras_mem.py cifar10 --no_augmentation --model_weights cifar10-quickdraw.h5 --test_only --npy | tee test-results-cifar10-after-quickdraw.txt
# python keras_mem.py cifar10 --no_augmentation --model_weights gimages-cifar10.h5 --test_only | tee test-results-cifar10-after-gimages.txt
# python keras_mem.py ./gimages-split  --model_weights gimages-cifar10.h5 --test_only | tee test-results-gimages-after-cifar10.txt
