import openai
import PyPDF2
import os
import pandas as pd
import time
filepath= "<LOCATION OF YOUR PDF FILE>"
openai.api_key  = "<YOUR OPENAI API KEY>"

def get_completion(prompt, model="gpt-3.5-turbo"):
  messages = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
     model=model,
     messages=messages,
     temperature=0, # this is the degree of randomness of the model's output
  )
  return response.choices[0].message["content"]

# creating a pdf file object
pdfFileObject = open(filepath, 'rb')
# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObject)
text=[]
summary=' '
#Storing the pages in a list
for i in range(0,len(pdfReader.pages)):
  # creating a page object
  pageObj = pdfReader.pages[i].extract_text()
  pageObj= pageObj.replace('\t\r','')
  pageObj= pageObj.replace('\xa0','')
  # extracting text from page
  text.append(pageObj)

  for i in range(len(text)):
  prompt =f"""
  Your task is to extract relevant information from a text on the page of a book. This information will be used to create a book summary.
  Extract relevant information from the following text, which is delimited with triple backticks.\
  Be sure to preserve the important details.
  Text: ```{text[i]}```
  """
  try:
    response = get_completion(prompt)
  except:
    response = get_completion(prompt)
  print(response)
  summary= summary+' ' +response +'\n\n'
  result.append(response)
  time.sleep(19)  #You can query the model only 3 times in a minute for free, so we need to put some delay

  with open('summary.txt', 'w') as out:
  out.write(summary)