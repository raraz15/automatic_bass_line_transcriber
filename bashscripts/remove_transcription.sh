# removes all the directories associated with the transcription process

for file in "data/outputs/*/"; do

    string="/F0_estimate"
    delete_string="$file$string"
    rm -rf $delete_string

    string="/midi"
    delete_string="$file$string"
    rm -rf $delete_string

    string="/pitch_track"
    delete_string="$file$string"
    rm -rf $delete_string    

    string="/quantized_pitch_track"
    delete_string="$file$string"
    rm -rf $delete_string  

    string="/exceptions/transciption"
    delete_string="$file$string"
    rm -rf $delete_string       

done