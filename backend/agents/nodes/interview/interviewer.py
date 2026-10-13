import asyncio
from agents.models.provider import chat_model
from agents.nodes.resume.extractor import extraction


extracted_content = asyncio.run(extraction()) #Filepath
print(extracted_content.content)
# print(extracted_content)
# print("summary:")
# print(extracted_content.summary)
# print("suspicious:")
# print(extracted_content.suspicious_content_detected)
