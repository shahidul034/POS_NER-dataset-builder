import pandas as pd
import nltk
nltk.download('punkt_tab')
def dataset_gen():
    from datasets import load_from_disk
    dataset = load_from_disk("data")
    return dataset

def already_completed_show(id):
    import pandas as pd
    import gradio as gr  # Assuming you're using Gradio
    import ast

    # Read the Excel file
    temp = pd.read_excel("data.xlsx")
    
    # Find the row(s) where the id matches
    row = temp[temp.isin([int(id)]).any(axis=1)]
    
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
                    <td>{pos_tag}</td>
                    <td>{ner_tag}</td>
                </tr>
            """
        
        # Close the table HTML
        html_table += "</tbody></table>"
        
        # Add HTML and Labels for each row
        output_elements.append(gr.HTML(value=f"<div style='overflow-x:auto;'>{html_table}</div>", visible=True))
        output_elements.append(gr.Label(value=" ".join(tokens), visible=True, label="Sentence"))
        output_elements.append(gr.Label(value=id, visible=True, label="Question No."))
    
    return output_elements



def display_table(front):
    file_path="data.xlsx"
    if os.path.exists(file_path):
        temp = pd.read_excel(file_path)
        print("File exists and has been loaded.")
    else:
        temp = pd.DataFrame(columns=['id', 'tokens', 'pos_tag', 'ner_tag'])
        temp.to_excel(file_path, index=False)
        
    if temp.isin([int(front)]).any().any()==True:
        gr.Warning("An answer already exists! If you submit another one, it will replace the previous answer.")
    
    df = pd.DataFrame(dataset[front])
    df=df.drop(['id','chunk_tags'],axis=1)
    text=" ".join(dataset[int(front)]['tokens'])
    df_with_custom_index = df.copy()
    df_with_custom_index.index = [f"Row {i+1}" for i in range(len(df))]
    
    # Convert DataFrame with custom index to HTML table
    html_table = df_with_custom_index.to_html(index=True)
    
    # Wrapping HTML table with Gradio Text
    # return f"<div style='overflow-x:auto;'>{html_table}</div>"
    return [gr.HTML(value=f"<div style='overflow-x:auto;'>{html_table}</div>", visible=True),gr.Label(value=text,visible=True,label="Sentence"),gr.Label(value=front,visible=True,label="Question No.")]

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

def show_tok(tran_text):
    import nltk
    # nltk.download('punkt')
    tokens = nltk.word_tokenize(tran_text)
    if tokens[len(tokens)-1]!="।":
        temp=tokens[len(tokens)-1]
        if temp[len(temp)-1]=="।":
            t1,t2=temp[:len(temp)-1],temp[len(temp)-1]
            tokens[len(tokens)-1]=t1
            tokens.append(t2)
        else:
            tokens.append("।")
    return {
                tok_text: gr.Label(value=f"{tokens}")
        }

def save_data(id, tok_text, pos_tag, ner_tag):
    import ast
   
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
"""
dataset=dataset_gen()
create_excel_if_not_exists()
front=int(open("current_data.txt","r").read())
text=" ".join(dataset[int(front)]['tokens'])
with gr.Blocks(css=css) as demo:
    with gr.Row():
        gr.Label(value="https://huggingface.co/datasets/conll2003",label="")
        gr.Label(value="Total: "+str(len(dataset)),label="")
    with gr.Row():
        ques=gr.Number(label="Question No.:")
        btn_ques=gr.Button("Move")
        show_btn=gr.Button("Show previous answer")
    with gr.Row():
        show_text1 = gr.HTML(visible=False) 
        with gr.Column():
            num_text=gr.Label(visible=False)
            show_text2 = gr.Label(visible=False)
    with gr.Row():
        tran_text=gr.Textbox(label="Enter translated text",info="sample of input: মিয়া বাপ্পি একদল বাঙালি কিশোরকে দেখিয়ে বলেন ।")
    with gr.Row():
        tok_text=gr.Label(label="Tokenized text")
    with gr.Row():
        pos_tag = gr.Textbox(label="Enter POS tag",info="sample of input: 22, 42, 16, 21, 35, 37, 16, 21, 7")
    with gr.Row():
        ner_tag = gr.Textbox(label="Enter NER tag",info="sample of input: 22, 42, 16, 21, 35, 37, 16, 21, 7")
    with gr.Row():
        lab_pos_ner = gr.HTML(visible=False)
    with gr.Row():
        check = gr.Button("Check")
        save = gr.Button("Save the data")
    with gr.Row():
        lab=gr.Label(visible=False)
    btn_ques.click(display_table,ques,[show_text1,show_text2,num_text])
    show_btn.click(already_completed_show,[ques],[show_text1,show_text2,num_text])
    tran_text.change(show_tok,tran_text,tok_text)
    check.click(pos_ner_show,[tok_text,pos_tag,ner_tag],[lab_pos_ner])
    # save.click(save_data,[num_text,tok_text,pos_tag,ner_tag],[lab,show_text1,show_text2])
    save.click(save_data,[num_text,tok_text,pos_tag,ner_tag],None)
    

demo.launch(share=False)