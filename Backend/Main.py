#from Api_key.key import MONGOURI
from Backend.Rules import NFLRule, parse_rules 
import pymongo

content=NFLRule()
parserules = parse_rules(content)


client = pymongo.MongoClient("{MONGOURI}")
db= client.QA_System
collection = db.NFL_Rules

results = collection.insert_many(parserules)
print(f"Inserted {len(results.inserted_ids)} documents")