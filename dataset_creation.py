import pandas as pd
import nltk
from tag_info import pos_tags_list,ner_tags_list,pos_tags_list_descr,ner_tags_list_descr,get_pos_tag_description,get_ner_tag_description
nltk.download('punkt_tab')

pos_tags_reverse = {v: k for k, v in pos_tags_list.items()}
ner_tags_reverse = {v: k for k, v in ner_tags_list.items()}



ban_map_ner={}
ban_map_pos={}


ner_tkn={}
pos_tkn={}


english_sentence="";










import spacy

# Load pre-trained language model
nlp_com = spacy.load("en_core_web_md")

def are_words_similar(word1, word2, threshold=0.7):
    # Process words using spaCy's NLP model
    token1 = nlp_com(word1)
    token2 = nlp_com(word2)
    
    # Calculate similarity (returns a float between 0 and 1)
    similarity = token1.similarity(token2)
    
    # Check if similarity exceeds the threshold
    return (similarity >= threshold),similarity





from deep_translator import GoogleTranslator

def translate_to_en(word):
    translator = GoogleTranslator(source='bn', target='en')
    translated = translator.translate(word)
    print(translated)
    
    return translated;

def translate_to_ban(word):
    translator = GoogleTranslator(source='en', target='bn')
    translated = translator.translate(word)
    print(translated)
    return translated;




def dataset_gen():
    from datasets import load_from_disk
    dataset = load_from_disk("data")
    return dataset




def already_completed_show(id):
    import pandas as pd
    import gradio as gr  
    import ast

    # Read the Excel file
    temp = pd.read_excel("data.xlsx")
    
    # Find the row(s) where the id matches
    row = temp[temp.isin([int(id)]).any(axis=1)]
    
    print("ROW--->",row)
    
    if len(row)==0:
        gr.Warning("Previous answer is not exist!!!")
    # Initialize a list to hold the outputs for each row
    output_elements = []
    
    # Iterate through each row of the DataFrame
    for i, row_data in row.iterrows():
        tokens = ast.literal_eval(row_data['tokens'])  # Convert string to list
        pos_tags = ast.literal_eval(row_data['pos_tag'])  # Convert string to list
        ner_tags = ast.literal_eval(row_data['ner_tag'])  # Convert string to list
        
        # Create an HTML table where each list element is a row in the table
        html_table = """
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr>
                    <th>Token</th>
                    <th>POS Tag</th>
                    <th>NER Tag</th>
                </tr>
            </thead>
            <tbody>
        """
        # Populate the table with each element from the tokens, pos_tags, and ner_tags lists
        for token, pos_tag, ner_tag in zip(tokens, pos_tags, ner_tags):
            html_table += f"""
                <tr>
                    <td>{token}</td>
                    <td>{pos_tag}({get_pos_tag_description(pos_tag)})</td>
                    <td>{ner_tag}({get_ner_tag_description(ner_tag)})</td>
                </tr>
            """
        
        # Close the table HTML
        html_table += "</tbody></table>"
        
        # Add HTML and Labels for each row
        output_elements.append(gr.HTML(value=f"<div style='overflow-x:auto;'>{html_table}</div>", visible=True))
        output_elements.append(gr.Label(value=" ".join(tokens), visible=True, label="Sentence"))
        output_elements.append(gr.Label(value=id, visible=True, label="Question No."))
    
    return output_elements










import json

with open("translation_data//banglat5_translations.json", "r", encoding="utf-8") as f1:
    bangla1_data = json.load(f1)

with open("translation_data//nllb_translations.json", "r", encoding="utf-8") as f2:
    bangla2_data = json.load(f2)








# Create a DataFrame with id, English, Bangla1, and Bangla2 columns
data = {
    "id": list(range(0, len(bangla1_data))),  # Create a range of IDs based on the number of translations
    "English": [item["en"] for item in bangla1_data],  # Use the English text from the bangla1_data
    "Bangla1": [item["bn"] for item in bangla1_data],  # Use the Bangla translation from banglat5_translations
    "Bangla2": [item["bn"] for item in bangla2_data]   # Use the Bangla translation from nllb_translations
}

df_translation = pd.DataFrame(data)






# Function to retrieve text and translations based on selected ID
def display_text(id):
    #print("Data ------>",data);
    
    print("------ Check -----DT")
    
    selected_row = df_translation[df_translation["id"] == id].iloc[0]  # Select the row where the id matches
    
    english_text = selected_row["English"]
    bangla1 = selected_row["Bangla1"].replace("<pad>","")
    bangla1 = bangla1.replace("</s>","")
    bangla2 = selected_row["Bangla2"]
    options = [bangla1, bangla2]
    return english_text, options


ban_df="";




def show_tok(tran_text,flag=1):
    global ban_df;
  ################################################################################################################################################  
    print("show_tok FN---------------------------->",tran_text,flag)
    
    import nltk
    # nltk.download('punkt')
    tokens = nltk.word_tokenize(tran_text)
    if tokens[len(tokens)-1]!="।":
        temp=tokens[len(tokens)-1]
        
        if temp[len(temp)-1]=="।":
            t1,t2=temp[:len(temp)-1],temp[len(temp)-1]
            tokens[len(tokens)-1]=t1
            tokens.append(t2)
        # else:
        #     tokens.append("।")
    
    print("show_tok FN END---------------------------->",tokens)    
    ban_df=tokens;
    
    
    ###################################################################
    
    #ner_sugg1,pos_sugg1,pos_tag1,ner_tag1,trans_google_text1=sugg()
    
    
    
    
    ##################################################################
    
    if flag==0:
        return tokens
    
    

    return {
                tok_text: gr.Label(value=f"{tokens}"),
                
        }
    
    
    
    

    
    



#############################################################################
eng_df=""




def display_table(sent_no,flag=1):
    global eng_df;
    
    print("display_table FN-->",dataset[sent_no])
    eng_df=dataset[sent_no];
    
    # translated_dataset = pd.read_excel("translated_dataset.xlsx")
    file_path="data.xlsx"
    if os.path.exists(file_path):
        temp = pd.read_excel(file_path)
        print("File exists and has been loaded.")
    else:
        temp = pd.DataFrame(columns=['id', 'tokens', 'pos_tags', 'ner_tags'])
        temp.to_excel(file_path, index=False)

    if temp.isin([int(sent_no)]).any().any()==True:
        gr.Warning("An answer already exists! If you submit another one, it will replace the previous answer.")
    
    
    df = pd.DataFrame(dataset[sent_no])
    #df['pos_tags'] = df['pos_tags'].apply(lambda x: f"{x}({get_pos_tag_description(x)})")
    #df['ner_tags'] = df['ner_tags'].apply(lambda x: f"{x}({get_ner_tag_description(x)})")
    df=df.drop(['id','chunk_tags'],axis=1)
    
    
    text=" ".join(dataset[int(sent_no)]['tokens'])
    df_with_custom_index = df.copy()
    df_with_custom_index.index = [f"Row {i+1}" for i in range(len(df))]
    if flag==0:
        return df_with_custom_index
    # Convert DataFrame with custom index to HTML table
    html_table = df_with_custom_index.to_html(index=True)
    # result=translated_dataset.loc[translated_dataset['id'] == front, 'Bangla']
    
    
    eng_text, options=display_text(sent_no)
    
    ##################################################################################
    #print("ENG sent-->",eng_text,options);
    ##################################################################################
    
    # if result.empty:
    #     result=""
    # else:
    #     result=result.values[0]
    
    print("-------------ck point------------>11.11")
    
    return [gr.HTML(value=f"<div style='overflow-x:auto;'>{html_table}</div>", visible=True),
    gr.Label(value=text,visible=True,label="Sentence"),
    gr.Label(value=sent_no,visible=True,label="Question No."), 
    gr.Radio(choices=options,label="Select Translation", visible=True,interactive=True)
    #gr.Textbox(label="Enter translated text",info="sample of input: MIA বাপ্পি একদল বাঙালি কিশোরকে দেখিয়ে বলেন ।",value=result)
               ]









import os
import pandas as pd

def create_excel_if_not_exists(filename='data.xlsx'):
    # Check if the file already exists
    if not os.path.exists(filename):
        # Create a new DataFrame with the specified columns
        df = pd.DataFrame(columns=['id', 'tokens', 'pos_tag', 'ner_tag'])
        
        # Save the DataFrame to an Excel file
        df.to_excel(filename, index=False)
        gr.Info(f"File '{filename}' created successfully with the columns: id, tokens, pos_tag, ner_tag.")
    else:
        gr.Info(f"File '{filename}' already exists.")









def pos_ner_show(tok_text,pos_tag,ner_tag):
    # Replace index names with custom labels
    print("------------------Text--------",tok_text)
    import ast
    tok_text = ast.literal_eval(tok_text)
    data={
        "tokens": tok_text,
        "pos_tag":pos_tag.split(","),
        'ner_tag':ner_tag.split(",")
    }
    try:
        df = pd.DataFrame(data)
    except:
        gr.Warning("All arrays must be of the same length!!!")
    df_with_custom_index = df.copy()
    df_with_custom_index.index = [f"Row {i+1}" for i in range(len(df))]
    
    # Convert DataFrame with custom index to HTML table
    html_table = df_with_custom_index.to_html(index=True)
    
    # Wrapping HTML table with Gradio Text
    return {lab_pos_ner: gr.Label(visible=True,value=f"<div style='overflow-x:auto;'>{html_table}</div>"),
        # lab: gr.Label(visible=True,elem_id="accepted",value="Submitted")
    }











def save_data(id, tok_text, pos_tag, ner_tag):
    import ast
    
    print("-------------ck point------------>2")
    print(tok_text)
    print(ban_df)
    
    print("-------------ck point------POS------>3")
    array_pos = pos_tag.split(",")
    print(pos_tag)
    
    
    print("-------------ck point------NER------>4")
    array_ner = ner_tag.split(",")
    print(ner_tag)
    
    # Read the Excel file
    file_path6 = 'token_data_map.xlsx'  # Replace with your file path
    df1 = pd.read_excel(file_path6, usecols="A:C")  # Read columns A and c
    
    
    ld=len(df1);
    for i in range(0,len(ban_df),1):
        xx=ban_df[i]
        if xx not in ban_map_ner:
            print("Token at save data FN-->",xx)
            df1.loc[ld] = [ban_df[i],array_ner[i],array_pos[i]]
            ld+=1
            file_path6 = 'token_data_map.xlsx'  # Replace with desired output path
            df1.to_excel(file_path6, index=False)
            
        
            
    
   
    # Read existing data from the Excel file
    df = pd.read_excel('data.xlsx')
    
    # Convert tok_text from string to list using ast.literal_eval
    tok_text2 = ast.literal_eval(tok_text)
    
    # Check if the id exists in the DataFrame
    exists = df['id'].isin([int(id)]).any()


    # Prepare new data to insert
    data = {
        'id': int(id),
        'tokens': tok_text,
        'pos_tag': pos_tag.split(","),
        'ner_tag': ner_tag.split(",")
    }



    # If the id exists, delete the row first
    if exists:
        df = df[df['id'] != int(id)]  # Remove the existing row where the id matches
        gr.Warning(f"Data with id {id} already exists, deleting the row and appending the new data.")

    # Append the new data to the DataFrame
    df = df._append(data, ignore_index=True)
    gr.Info("New data appended successfully!")

    # Save the updated DataFrame back to the Excel file
    df.to_excel('data.xlsx', index=False)









import gradio as gr
css = """
#accepted {background-color: green;align-content: center;font: 30px Arial, sans-serif;}
#wrong {background-color: red;align-content: center;font: 30px Arial, sans-serif;}
#already {background-color: blue;align-content: center;font: 30px Arial, sans-serif;}
#ner_sugg_label { align-content: center;font: 15px Arial, sans-serif; }
"""
dataset=dataset_gen()
create_excel_if_not_exists()
# sent_no=int(open("current_data.txt","r").read())
# text=" ".join(dataset[int(front)]['tokens'])
with gr.Blocks(css=css) as demo:
    with gr.Row():
        gr.Label(value="https://huggingface.co/datasets/conll2003",label="Dataset link")
        gr.Label(value="Total: "+str(len(dataset)),label="Total number of sentence in dataset"),
        gr.Label(value="Completed: "+str(len(pd.read_excel("data.xlsx"))),label="Total number of sentence completed")
  
  
  
        
    with gr.Row():
        ques=gr.Number(label="Question No.:")
        btn_ques=gr.Button("Move")
        show_btn=gr.Button("Show previous answer")
        suggest_btn=gr.Button("Auto Suggest")
        
        
        
        
        
        
    with gr.Row():
        show_text1 = gr.HTML(visible=False) 
        with gr.Column():
            num_text=gr.Label(visible=False)
            show_text2 = gr.Label(visible=False)
            translation_options = gr.Radio(visible=False)
            
        
        
    ##################################################################################
    with gr.Row():
        trans_google_text=gr.Label(label="Translated By Google")
        
            
    with gr.Row():
        tran_text=gr.Textbox(label="Enter translated text",info="sample of input: মিয়া বাপ্পি একদল বাঙালি কিশোরকে দেখিয়ে বলেন ।")
        
        
        
    with gr.Row():
        tok_text=gr.Label(label="Tokenized text")
        #print("Tokenized text--->",tok_text)
    
    
 #tran_text.change(show_tok,tran_text,tok_text)  
    
    with gr.Row():
        pos_sugg=gr.Label(label="POS Suggesion")
        
        
    with gr.Row():
        pos_tag = gr.Textbox(label="Enter POS tag",info="sample of input: 22, 42, 16, 21, 35, 37, 16, 21, 7")
    
    
    #with gr.Row():
        #ner_sugg=gr.Label(label="NER Suggesion")
        
        
    with gr.Row():
        ner_sugg = gr.Label(label="NER Suggestion", elem_id="ner_sugg_label")  # Use elem_id to apply custom CSS
        
        
    with gr.Row():
        ner_tag = gr.Textbox(label="Enter NER tag",info="sample of input: 22, 42, 16, 21, 35, 37, 16, 21, 7")
    with gr.Row():
        lab_pos_ner = gr.HTML(visible=False)
    
    
    with gr.Row():
        check = gr.Button("Check")
        save = gr.Button("Save the data")
    with gr.Row():
        lab=gr.Label(visible=False)
        ########################################################################
        
        
        
        
        
    with gr.Row():
        gr.Markdown(
            """
## POS Tags Documentation

- `"` : 0 - Quotation mark (open or close)
- `''` : 1 - Closing quotation mark
- `#` : 2 - Number sign (e.g., hashtags)
- `$` : 3 - Dollar sign
- `(` : 4 - Left parenthesis
- `)` : 5 - Right parenthesis
- `,` : 6 - Comma
- `।` : 7 - Bengali punctuation mark for a full stop (period)
- `:` : 8 - Colon
- ``` ` ``` : 9 - Opening quotation mark
- `CC` : 10 - Coordinating conjunction (e.g., *and*, *but*)
- `CD` : 11 - Cardinal number (e.g., *one*, *two*)
- `DT` : 12 - Determiner (e.g., *the*, *a*)
- `EX` : 13 - Existential "there" (e.g., *there is*)
- `FW` : 14 - Foreign word
- `IN` : 15 - Preposition or subordinating conjunction (e.g., *in*, *on*, *because*)
- `JJ` : 16 - Adjective (e.g., *big*, *old*)
- `JJR` : 17 - Adjective, comparative (e.g., *bigger*, *older*)
- `JJS` : 18 - Adjective, superlative (e.g., *biggest*, *oldest*)
- `LS` : 19 - List item marker (e.g., *1.*, *2.*)
- `MD` : 20 - Modal verb (e.g., *can*, *should*)
- `NN` : 21 - Noun, singular or mass (e.g., *cat*, *laughter*)
- `NNP` : 22 - Proper noun, singular (e.g., *John*, *Paris*)
- `NNPS` : 23 - Proper noun, plural (e.g., *Americans*)
- `NNS` : 24 - Noun, plural (e.g., *cats*)
- `NN|SYM` : 25 - Noun or symbol (rare case)
- `PDT` : 26 - Predeterminer (e.g., *all*, *both*)
- `POS` : 27 - Possessive ending (e.g., *'s*)
- `PRP` : 28 - Personal pronoun (e.g., *I*, *he*, *they*)
- `PRP$` : 29 - Possessive pronoun (e.g., *my*, *his*, *their*)
- `RB` : 30 - Adverb (e.g., *quickly*, *never*)
- `RBR` : 31 - Adverb, comparative (e.g., *faster*)
- `RBS` : 32 - Adverb, superlative (e.g., *fastest*)
- `RP` : 33 - Particle (e.g., *up*, *off*)
- `SYM` : 34 - Symbol (e.g., math or currency symbols)
- `TO` : 35 - "To" as a preposition or infinitive marker (e.g., *to go*)
- `UH` : 36 - Interjection (e.g., *uh*, *well*)
- `VB` : 37 - Verb, base form (e.g., *run*, *eat*)
- `VBD` : 38 - Verb, past tense (e.g., *ran*, *ate*)
- `VBG` : 39 - Verb, gerund or present participle (e.g., *running*, *eating*)
- `VBN` : 40 - Verb, past participle (e.g., *run*, *eaten*)
- `VBP` : 41 - Verb, non-3rd person singular present (e.g., *I run*)
- `VBZ` : 42 - Verb, 3rd person singular present (e.g., *he runs*)
- `WDT` : 43 - Wh-determiner (e.g., *which*, *that*)
- `WP` : 44 - Wh-pronoun (e.g., *who*, *what*)
- `WP$` : 45 - Possessive wh-pronoun (e.g., *whose*)
- `WRB` : 46 - Wh-adverb (e.g., *how*, *where*)

---

## NER Tags Documentation

- `O` : 0 - Outside of any named entity
- `B-ADJP` : 1 - Beginning of an adjective phrase
- `I-ADJP` : 2 - Inside an adjective phrase
- `B-ADVP` : 3 - Beginning of an adverb phrase
- `I-ADVP` : 4 - Inside an adverb phrase
- `B-CONJP` : 5 - Beginning of a conjunction phrase
- `I-CONJP` : 6 - Inside a conjunction phrase
- `B-INTJ` : 7 - Beginning of an interjection
- `I-INTJ` : 8 - Inside an interjection
- `B-LST` : 9 - Beginning of a list item
- `I-LST` : 10 - Inside a list item
- `B-NP` : 11 - Beginning of a noun phrase (e.g., *the big dog*)
- `I-NP` : 12 - Inside a noun phrase
- `B-PP` : 13 - Beginning of a prepositional phrase (e.g., *in the house*)
- `I-PP` : 14 - Inside a prepositional phrase
- `B-PRT` : 15 - Beginning of a particle phrase (e.g., *up* in *give up*)
- `I-PRT` : 16 - Inside a particle phrase
- `B-SBAR` : 17 - Beginning of a subordinate clause (e.g., *that he left*)
- `I-SBAR` : 18 - Inside a subordinate clause
- `B-UCP` : 19 - Beginning of an unlike coordinated phrase
- `I-UCP` : 20 - Inside an unlike coordinated phrase
- `B-VP` : 21 - Beginning of a verb phrase (e.g., *ran quickly*)
- `I-VP` : 22 - Inside a verb phrase


"""

        )
        
        
        
        
        
    btn_ques.click(display_table,ques,[show_text1,show_text2,num_text,translation_options])
    # Define a wrapper function
    #def display_hello(sent_no):
    #   print("hello")  # Print "hello" to the console when the button is clicked
    #  print("CK---->",show_text1, show_text2, num_text, translation_options)
    #   return display_table(sent_no)  # Call the original display_table function

    # Modify the button click behavior to call the new wrapper function
    #btn_ques.click(display_hello, ques, [show_text1, show_text2, num_text, translation_options])


    
    
    def translation_change(translation_options,sent_no):
        english_text, options=display_text(sent_no)
        
        
        df_index=display_table(sent_no,0)
        #print("ENG--BAN sent-->",english_text,options);
        print("ENG sent-->",english_text);
        
        global english_sentence;
        english_sentence=english_text;
        
        
        
        tok_text=show_tok(translation_options,0)
       
        
        
        
        if len(df_index)>len(tok_text):
            for i in range(len(df_index)-len(tok_text)):
                tok_text.append("Unknown")
                
        elif len(df_index)<len(tok_text):
            tok_text=tok_text[:len(df_index)]
            
        
        
        print("Tokens-->",tok_text);
        
        
        
            
        df_index["Probable Bangla Token"]=tok_text
        html_table = df_index.to_html(index=True)
        
        
        return [gr.Textbox(label="Enter translated text",info="sample of input: মিয়া বাপ্পি একদল বাঙালি কিশোরকে দেখিয়ে বলেন ।",value=translation_options),
                gr.HTML(value=f"<div style='overflow-x:auto;'>{html_table}</div>", visible=True)
                ]





    translation_options.change(translation_change,[translation_options,ques], [tran_text,show_text1])
    
    
    show_btn.click(already_completed_show,[ques],[show_text1,show_text2,num_text])
    
    
    
    
    
    
    
    
    def sugg():
        
        # Read the Excel file
        file_path6 = 'token_data_map.xlsx'  # Replace with your file path
        df11 = pd.read_excel(file_path6, usecols="A:C")  # Read columns A and c
        
        len_b_xl=len(df11)
        for i in range(0,len_b_xl,1):
            aa=df11.iloc[i, :]
            ban_map_ner[aa["token"]]=aa["ner"]
            #ban_map_pos[aa["token"]]=aa["pos"]

       

        
        global ban_df
        global eng_df
        print(ban_df)
        print(eng_df)
        
        eng_tkn_list = eng_df["tokens"]
        eng_ner_tkn_list = eng_df["ner_tags"]
        eng_pos_tkn_list = eng_df["pos_tags"]
        
        #print("Token-->",eng_tkn_list[1],eng_pos_tkn_list[1])
        
        for i in range(0,len(eng_tkn_list),1):
            ner_tkn[eng_tkn_list[i].lower()]=eng_ner_tkn_list[i]
            pos_tkn[eng_tkn_list[i].lower()]=eng_pos_tkn_list[i]
            
        
        str_nar=""
        str_pos=""
        
        str_nar_sug=""
        str_pos_sug=""
        
        for xx in ban_df:
            
            
               #print("----------------------DF--------------------->",ban_df);
               #print("---------------------------Error in---------------------->",xx);
 
               tk=translate_to_en(xx)
               #print("---------------------------Bangla Tran---------------------->",xx);
               #tk = tk.lower()
               
                   
               if tk is not None and isinstance(tk, str):
                     if tk.isalpha():
                          tk = tk.lower()  # Convert to lowercase only if tk is a word (contains only letters)
    

               
               
               
               print(xx,"trn-->",tk)
               
               if tk in ner_tkn:
                   str_nar+=str(ner_tkn[tk])+","
                   str_pos+=str(pos_tkn[tk])+","
                   
         
                   
               elif xx in ban_map_ner:
                   str_nar+=str(ban_map_ner[xx])+","
                   #str_pos+=str(ban_map_pos[xx])+","
                   str_pos+="#,"
                   str_pos_sug+=xx+","
                
               else:
                  
                 if tk is not None and isinstance(tk, str): 
                  for tkn in eng_tkn_list:
                      if(tkn=="."):continue;
                      check,score=are_words_similar(tk, tkn)
                      print("Main--->",tk,"--->",tkn,"-->Score--->",score)
                      
                 ####################################################  
                 str_nar+="#,"
                 str_pos+="#,"
                 
                 str_nar_sug+=xx+","
                 str_pos_sug+=xx+","
                 
                 
               
        print("Str  NAR--->",str_nar)
        print("Str  POS--->",str_pos)
        
        str_nar = str_nar[:-1]
        str_pos = str_pos[:-1]
        
        str_nar_sug = str_nar_sug[:-1]
        str_pos_sug = str_pos_sug[:-1]
        
        
        tr_g=translate_to_ban(english_sentence)
        
        return {
                ner_sugg: gr.Label(value=str_nar_sug),
                pos_sugg: gr.Label(value=str_pos_sug),
                pos_tag:gr.Label(value=str_pos),
                ner_tag: gr.Label(value=str_nar),
                trans_google_text: gr.Label(value=tr_g)
                
        }
        
        
        
    def sugg_2():
        
        # Read the Excel file
        file_path6 = 'token_data_map.xlsx'  # Replace with your file path
        df11 = pd.read_excel(file_path6, usecols="A:C")  # Read columns A and c
        
        len_b_xl=len(df11)
        for i in range(0,len_b_xl,1):
            aa=df11.iloc[i, :]
            ban_map_ner[aa["token"]]=aa["ner"]
            #ban_map_pos[aa["token"]]=aa["pos"]

       

        
        global ban_df
        global eng_df
        print(ban_df)
        print(eng_df)
        
        eng_tkn_list = eng_df["tokens"]
        eng_ner_tkn_list = eng_df["ner_tags"]
        eng_pos_tkn_list = eng_df["pos_tags"]
        
        #print("Token-->",eng_tkn_list[1],eng_pos_tkn_list[1])
        
        for i in range(0,len(eng_tkn_list),1):
            ner_tkn[eng_tkn_list[i].lower()]=eng_ner_tkn_list[i]
            pos_tkn[eng_tkn_list[i].lower()]=eng_pos_tkn_list[i]
            
        
        str_nar=""
        str_pos=""
        
        str_nar_sug=""
        str_pos_sug=""
        
        for xx in ban_df:
 
               tk=translate_to_en(xx)
               #tk = tk.lower()
               if tk is not None and isinstance(tk, str):
                     if tk.isalpha():
                          tk = tk.lower()
               
               
               
               print(xx,"trn-->",tk)
               
               if tk in ner_tkn:
                   str_nar+=str(ner_tkn[tk])+","
                   str_pos+=str(pos_tkn[tk])+","
                   
         
                   
               elif xx in ban_map_ner:
                   str_nar+=str(ban_map_ner[xx])+","
                   #str_pos+=str(ban_map_pos[xx])+","
                   str_pos+="#,"
                   str_pos_sug+=xx+","
                
               else:
                
                 if tk is not None and isinstance(tk, str):
                  for tkn in eng_tkn_list:
                      if(tkn=="."):continue;
                      check,score=are_words_similar(tk, tkn)
                      print("Main--->",tk,"--->",tkn,"-->Score--->",score)
                      
                 ####################################################  
                 str_nar+="#,"
                 str_pos+="#,"
                 
                 str_nar_sug+=xx+","
                 str_pos_sug+=xx+","
                 
                 
               
        print("Str  NAR--->",str_nar)
        print("Str  POS--->",str_pos)
        
        str_nar = str_nar[:-1]
        str_pos = str_pos[:-1]
        str_nar_sug = str_nar_sug[:-1]
        str_pos_sug = str_pos_sug[:-1]
        
        
        return {
                ner_sugg: gr.Label(value=str_nar_sug),
                pos_sugg: gr.Label(value=str_pos_sug),
                pos_tag:gr.Label(value=str_pos),
                ner_tag: gr.Label(value=str_nar),  
        }
        
        
        
        
    
        
    
    ###################
    suggest_btn.click(sugg,[],[ner_sugg,pos_sugg,pos_tag,ner_tag,trans_google_text])




    tran_text.change(show_tok,tran_text,tok_text)
    
    
    tok_text.change(sugg_2,[],[ner_sugg,pos_sugg,pos_tag,ner_tag])
    
    
  
    
    #print("noW---111>",show_tok,tran_text,tok_text)
    

    
    
    ########################################################################################
    check.click(pos_ner_show,[tok_text,pos_tag,ner_tag],[lab_pos_ner])
    
    
    
    
    # save.click(save_data,[num_text,tok_text,pos_tag,ner_tag],[lab,show_text1,show_text2])
    save.click(save_data,[num_text,tok_text,pos_tag,ner_tag],None)
    
    
    
    
    

demo.launch(share=False)