# removes all the directories associated with the extraction process
# data/outputs/track_title/

for file in "data/outputs/*/"; do

    string="/bass_line"
    delete_string="$file$string"
    rm -rf $delete_string

    string="/beat_grid"
    delete_string="$file$string"
    rm -rf $delete_string

    string="/chorus"
    delete_string="$file$string"
    rm -rf $delete_string  

    string="/exceptions/extraction"
    delete_string="$file$string"
    rm -rf $delete_string         

done