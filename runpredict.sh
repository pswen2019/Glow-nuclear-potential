model='--model ./trainedGlow/final_model.pth --modelconfig ./trainedGlow/config.json'

vitn2model="--vitchiraln2model ./vitmodel/chiral48/chiraln2/vit_chiraln2lo.pth"
vitn2norc="--vitchiraln2normc ./vitmodel/chiral48/chiraln2/vit_chiraln2lo_normc"
vitn2config="--vitchiraln2config ./vitmodel/chiral48/chiraln2/vit_chiraln2lo_config.json"

vitn3model="--vitchiraln3model ./vitmodel/chiral48/chiraln3/vit_chiraln3lo.pth"
vitn3norc="--vitchiraln3normc ./vitmodel/chiral48/chiraln3/vit_chiraln3lo_normc"
vitn3config="--vitchiraln3config ./vitmodel/chiral48/chiraln3/vit_chiraln3lo_config.json"

python ./predict.py $model $vitn2model $vitn2config $vitn2norc $vitn3model $vitn3config $vitn3norc
