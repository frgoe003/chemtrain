python3 jax/tools/jax_to_ir.py \
--fn prog.fn \
--input_shapes '[("position", "f32[5,3]"), ("neighbor_idx", "s32[5,4]")]' \
--ir_human_dest /tmp/fn_hlo.txt \
--ir_format HLO \
--ir_dest /tmp/fn_hlo.pb
