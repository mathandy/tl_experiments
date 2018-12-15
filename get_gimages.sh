name=gimages_simple
googleimagesdownload -cf gimages_config/"${name}"_config.json | tee gimages_config/"${name}"_urls.txt
mv downloads "${name}"
mkdir "${name}"-split
mkdir "${name}"-split/train
mkdir "${name}"-split/val
cp -r "${name}"/* "${name}"-split/train
python split_dataset.py "${name}"-split
python make_gray.py "${name}"-split "${name}"-split-gray
