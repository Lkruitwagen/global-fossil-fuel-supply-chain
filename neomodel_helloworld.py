from neomodel import StructuredNode, StringProperty, RelationshipTo, RelationshipFrom, config, db

config.DATABASE_URL = 'bolt://neo4j:neo4jpass@localhost:7687'
#config.ENCRYPTED_CONNECTION=False


# clear the database
results, meta = db.cypher_query("MATCH (n) DETACH DELETE n;", {})

class Book(StructuredNode):
    title = StringProperty(unique_index=True)
    author = RelationshipTo('Author', 'AUTHOR')

class Author(StructuredNode):
    name = StringProperty(unique_index=True)
    books = RelationshipFrom('Book', 'AUTHOR')

lefthand = Book(title='Left Hand of Darkness').save()
ursula =  Author(name='ursula  K le guin').save()
lefthand.author.connect(ursula)