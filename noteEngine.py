from llama_index.core.tools import FunctionTool
import os

notes_file= os.path.join("notes_data", "notes_file")

print("Notes file path")
print(notes_file)


def save_notes(notes):
 if not os.path.exists(notes_file):
   open(notes_file,"w")


 with open(notes_file,"a") as f:
    f.writelines([notes + "\n"])

 return "notes saved"

noteEngine=FunctionTool.from_defaults(
  fn=save_notes,
  name="note_saver",
  description="This tool can save the text based note for the user"  
)  


 