# removes all the directories associated with the extraction process
# data/outputs/

for file in "data/outputs/*/"; do

    rm -rf $file

done