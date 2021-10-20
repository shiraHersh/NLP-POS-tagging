# NLP-POS-tagging
List of features:

  Basic features:
  1. < tag_i >
  2. < w_i, tag_i >
  3. < tag_i-1, tag_i >
  4. < tag_i-2, tag_i-1, tag_i >
  5. < prefix_len_3, tag_i >
  6. < suffix_len_3, tag_i >


  Word shapes: 
  * A-Z = X
  * a-z = x
  * 0-9 = d
  
  Examples:
  * full_word_shape(Google3d.INC) = Xxxxxxdx.XXX
  * small_word_shape(Google3d.INC) = Xxdx.X
  
  
  Advanced features:
  
  1. < tag_i, w_contain_capital >
  2. < tag_i, w_contain_number >
  3. < tag_i, w_full_shape >
  4. < tag_i, w_small_shape >
  
  5. < tag_i, w_i-1 >
  6. < tag_i, w_i+1 >
  7. < tag_i, tag_i+1 >
  8. < tag_i, tag_i+1, tag_i+2 >
  9. < tag_i, tag_i-1, tag_i+1 >

  10. < prefix_len_2, tag_i >
  11. < suffix_len_2, tag_i >
  12. < prefix_len_1, tag_i >
  13. < suffix_len_1, tag_i >
