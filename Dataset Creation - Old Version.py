# Import necessary libraries
import pandas as pd  # For data manipulation and working with Excel
import nltk  # For natural language processing tasks
nltk.download('punkt_tab')  # Download NLTK tokenizer data

# Function to load the dataset from disk using Hugging Face datasets library
def dataset_gen():
    from datasets import load_from_disk  # Import inside function to prevent global dependencies
    dataset = load_from_disk("data")  # Load the dataset from the local "data" folder
    return dataset  # Return the loaded dataset

# Function to show already completed question answers from an Excel file
def already_completed_show(id):
    import pandas as pd
    import gradio as gr  # Assuming you're using Gradio for user interface
    import ast  # For converting string representations of lists back into lists

    # Read the Excel file containing the data
    temp = pd.read_excel("data.xlsx")
    
    # Find the row(s) where the provided id matches any value in the dataset
    row = temp[temp.isin([int(id)]).any(axis=1)]
    if len(row) == 0:
        gr.Warning("Previous answer does not exist!!!")  # If no matching row found, show a warning
    
    # Initialize a list to store output elements (e.g., HTML, Labels)
    output_elements = []
    
    # Iterate through each row in the DataFrame to display the tokens, POS tags, and NER tags
    for i, row_data in row.iterrows():
        # Convert string representations of lists back to actual lists
        tokens = ast.literal_eval(row_data['tokens'])  
        pos_tags = ast.literal_eval(row_data['pos_tag'])
        ner_tags = ast.literal_eval(row_data['ner_tag'])
        
        # Create an HTML table to display the tokens, POS tags, and NER tags
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
        # Populate the table with the tokens, POS tags, and NER tags
        for token, pos_tag, ner_tag in zip(tokens, pos_tags, ner_tags):
            html_table += f"""
                <tr>
                    <td>{token}</td>
                    <td>{pos_tag}</td>
                    <td>{ner_tag}</td>
                </tr>
            """
        html_table += "</tbody></table>"  # Close the HTML table
        
        # Append the HTML table and other elements to the output
        output_elements.append(gr.HTML(value=f"<div style='overflow-x:auto;'>{html_table}</div>", visible=True))
        output_elements.append(gr.Label(value=" ".join(tokens), visible=True, label="Sentence"))
        output_elements.append(gr.Label(value=id, visible=True, label="Question No."))
    
    return output_elements  # Return the output elements to be displayed in the interface

# Load translations from two different models: BanglaT5 and NLLB
import json

# Load translation data from JSON files
with open("translation_data//banglat5_translations.json", "r", encoding="utf-8") as f1:
    bangla1_data = json.load(f1)

with open("translation_data//nllb_translations.json", "r", encoding="utf-8") as f2:
    bangla2_data = json.load(f2)

# Create a DataFrame with id, English, and two Bangla translations
data = {
    "id": list(range(0, len(bangla1_data))),  # Create an ID range based on the number of translations
    "English": [item["en"] for item in bangla1_data],  # Extract English text from bangla1_data
    "Bangla1": [item["bn"] for item in bangla1_data],  # Extract Bangla1 translations from bangla1_data
    "Bangla2": [item["bn"] for item in bangla2_data]   # Extract Bangla2 translations from bangla2_data
}

df_translation = pd.DataFrame(data)  # Convert the data into a DataFrame

# Function to display text and translation options for a given ID
def display_text(id):
    selected_row = df_translation[df_translation["id"] == id].iloc[0]  # Get the row that matches the ID
    english_text = selected_row["English"]  # Retrieve the English text
    bangla1 = selected_row["Bangla1"].replace("<pad>", "").replace("</s>", "")  # Clean up Bangla1 text
    bangla2 = selected_row["Bangla2"]  # Retrieve Bangla2 text
    options = [bangla1, bangla2]  # Provide the Bangla translation options
    return english_text, options  # Return the English text and translation options

# Function to tokenize translated text and ensure it ends with Bangla punctuation
def show_tok(tran_text, flag=1):
    import nltk
    tokens = nltk.word_tokenize(tran_text)  # Tokenize the translated text using NLTK
    
    if tokens[-1] != "।":  # Check if the last token is not the Bangla punctuation '।'
        temp = tokens[-1]
        if temp[-1] == "।":  # Split the last token if it ends with '।'
            t1, t2 = temp[:-1], temp[-1]
            tokens[-1] = t1
            tokens.append(t2)
        else:
            tokens.append("।")  # Add the '।' if it is missing
    if flag == 0:
        return tokens  # Return the tokens
    return {
        tok_text: gr.Label(value=f"{tokens}")  # Display the tokens in Gradio interface
    }

# Function to display the dataset and translated sentence
def display_table(sent_no, flag=1):
    file_path = "data.xlsx"
    
    # Check if the Excel file exists
    if os.path.exists(file_path):
        temp = pd.read_excel(file_path)
        print("File exists and has been loaded.")
    else:
        # Create an empty DataFrame if the file doesn't exist
        temp = pd.DataFrame(columns=['id', 'tokens', 'pos_tag', 'ner_tag'])
        temp.to_excel(file_path, index=False)

    # Check if the sentence number is already present in the Excel file
    if temp.isin([int(sent_no)]).any().any():
        gr.Warning("An answer already exists! If you submit another one, it will replace the previous answer.")
    
    # Fetch the sentence data from the dataset and remove unnecessary columns
    df = pd.DataFrame(dataset[sent_no]).drop(['id', 'chunk_tags'], axis=1)
    text = " ".join(dataset[int(sent_no)]['tokens'])  # Join the tokens to form a sentence
    
    # Modify the DataFrame to have a custom index
    df_with_custom_index = df.copy()
    df_with_custom_index.index = [f"Row {i+1}" for i in range(len(df))]
    
    # If flag is set to 0, return the DataFrame
    if flag == 0:
        return df_with_custom_index
    
    # Convert the DataFrame to HTML for display
    html_table = df_with_custom_index.to_html(index=True)
    eng_text, options = display_text(sent_no)
    
    # Return Gradio components for displaying the sentence and options
    return [
        gr.HTML(value=f"<div style='overflow-x:auto;'>{html_table}</div>", visible=True),
        gr.Label(value=text, visible=True, label="Sentence"),
        gr.Label(value=sent_no, visible=True, label="Question No."),
        gr.Radio(choices=options, label="Select Translation", visible=True, interactive=True)
    ]


import os
import pandas as pd

# Function to create the Excel file if it doesn't exist
def create_excel_if_not_exists(filename='data.xlsx'):
    # Check if the file already exists
    if not os.path.exists(filename):
        # Create a new DataFrame with specified columns
        df = pd.DataFrame(columns=['id', 'tokens', 'pos_tag', 'ner_tag'])
        df.to_excel(filename, index=False)  # Save it as an Excel file
        gr.Info(f"File '{filename}' created successfully with the columns: id, tokens, pos_tag, ner_tag.")
    else:
        gr.Info(f"File '{filename}' already exists.")  # Inform that the file already exists

# Function to display POS and NER tags along with tokens
def pos_ner_show(tok_text, pos_tag, ner_tag):
    import ast
    tok_text = ast.literal_eval(tok_text)  # Convert token text back into a list
    data = {
        "tokens": tok_text,
        "pos_tag": pos_tag.split(","),  # Split POS tags into a list
        'ner_tag': ner_tag.split(",")   # Split NER tags into a list
    }
    try:
        df = pd.DataFrame(data)  # Create a DataFrame from the data
    except:
        gr.Warning("All arrays must be of the same length!!!")  # Ensure that all lists are of equal length
    
    # Create a copy with custom row labels
    df_with_custom_index = df.copy()
    df_with_custom_index.index = [f"Row {i+1}" for i in range(len(df))]
    
    # Convert the DataFrame to an HTML table for display
    html_table = df_with_custom_index.to_html(index=True)
    
    # Return the HTML table to Gradio for display
    return {
        lab_pos_ner: gr.Label(visible=True, value=f"<div style='overflow-x:auto;'>{html_table}</div>")
    }

# Function to save tokenized data, POS tags, and NER tags to the Excel file
def save_data(id, tok_text, pos_tag, ner_tag):
    import ast
    
    # Read the existing data from the Excel file
    df = pd.read_excel('data.xlsx')
    
    # Convert token text from string to list
    tok_text2 = ast.literal_eval(tok_text)
    
    # Check if the ID already exists in the DataFrame
    exists = df['id'].isin([int(id)]).any()

    # Prepare the new data for insertion
    data = {
        'id': int(id),
        'tokens': tok_text,
        'pos_tag': pos_tag.split(","),
        'ner_tag': ner_tag.split(",")
    }

    # If the ID exists, remove the existing row
    if exists:
        df = df[df['id'] != int(id)]  # Remove the existing row
        gr.Warning(f"Data with id {id} already exists, deleting the row and appending the new data.")

    # Append the new data to the DataFrame
    df = df._append(data, ignore_index=True)
    gr.Info("New data appended successfully!")  # Inform that the data was successfully added

    # Save the updated DataFrame back to the Excel file
    df.to_excel('data.xlsx', index=False)  # Save it back to the Excel file

# Gradio Interface (Frontend)
import gradio as gr
css = """
#accepted {background-color: green;align-content: center;font: 30px Arial, sans-serif;}
#wrong {background-color: red;align-content: center;font: 30px Arial, sans-serif;}
#already {background-color: blue;align-content: center;font: 30px Arial, sans-serif;}
"""
dataset = dataset_gen()  # Load the dataset
create_excel_if_not_exists()  # Ensure the Excel file exists

# Gradio app layout
with gr.Blocks(css=css) as demo:
    # Row showing dataset link and counts
    with gr.Row():
        gr.Label(value="https://huggingface.co/datasets/conll2003", label="Dataset link")
        gr.Label(value="Total: " + str(len(dataset)), label="Total number of sentences in dataset"),
        gr.Label(value="Completed: " + str(len(pd.read_excel("data.xlsx"))), label="Total number of sentences completed")
    
    # Input for question number and navigation buttons
    with gr.Row():
        ques = gr.Number(label="Question No.:")
        btn_ques = gr.Button("Move")
        show_btn = gr.Button("Show previous answer")
    
    # Show previous answer area
    with gr.Row():
        show_text1 = gr.HTML(visible=False) 
        with gr.Column():
            num_text = gr.Label(visible=False)
            show_text2 = gr.Label(visible=False)
            translation_options = gr.Radio(visible=False)
    
    # Input for translated text and tokenized text
    with gr.Row():
        tran_text = gr.Textbox(label="Enter translated text", info="sample of input: মিয়া বাপ্পি একদল বাঙালি কিশোরকে দেখিয়ে বলেন ।")
    with gr.Row():
        tok_text = gr.Label(label="Tokenized text")
    
    # Inputs for POS and NER tags
    with gr.Row():
        pos_tag = gr.Textbox(label="Enter POS tag", info="sample of input: 22, 42, 16, 21, 35, 37, 16, 21, 7")
    with gr.Row():
        ner_tag = gr.Textbox(label="Enter NER tag", info="sample of input: 22, 42, 16, 21, 35, 37, 16, 21, 7")
    
    # POS and NER table display
    with gr.Row():
        lab_pos_ner = gr.HTML(visible=False)
    
    # Buttons for checking and saving data
    with gr.Row():
        check = gr.Button("Check")
        save = gr.Button("Save the data")
    with gr.Row():
        lab = gr.Label(visible=False)

    # Function calls for buttons
    btn_ques.click(display_table, ques, [show_text1, show_text2, num_text, translation_options])
    def translation_change(translation_options, sent_no):
        english_text, options = display_text(sent_no)
        df_index = display_table(sent_no, 0)
        tok_text = show_tok(translation_options, 0)
        if len(df_index) > len(tok_text):
            for i in range(len(df_index) - len(tok_text)):
                tok_text.append("Unknown")
        elif len(df_index) < len(tok_text):
            tok_text = tok_text[:len(df_index)]
        df_index["Probable Bangla Token"] = tok_text
        html_table = df_index.to_html(index=True)
        return [gr.Textbox(label="Enter translated text", info="sample of input: মিয়া বাপ্পি একদল বাঙালি কিশোরকে দেখিয়ে বলেন ।", value=translation_options),
                gr.HTML(value=f"<div style='overflow-x:auto;'>{html_table}</div>", visible=True)]
    
    # Link functions to Gradio components
    translation_options.change(translation_change, [translation_options, ques], [tran_text, show_text1])
    show_btn.click(already_completed_show, [ques], [show_text1, show_text2, num_text])
    tran_text.change(show_tok, tran_text, tok_text)
    check.click(pos_ner_show, [tok_text, pos_tag, ner_tag], [lab_pos_ner])
    save.click(save_data, [num_text, tok_text, pos_tag, ner_tag], None)

# Launch Gradio app
demo.launch(share=False)
