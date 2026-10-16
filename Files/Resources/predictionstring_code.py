# The CreatePredictionstring function here was used to create the predictionstring column
def CreatePredictionstring(full_text, discourse_start, discourse_end):
    char_start = discourse_start
    char_end = discourse_end
    word_start = len(full_text[:char_start].split())
    word_end = word_start + len(full_text[char_start:char_end].split())
    word_end = min( word_end, len(full_text.split()) )
    predictionstring = [x for x in range(word_start,word_end)]
    return predictionstring