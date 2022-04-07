import mysql.connector

mydb = mysql.connector.connect(user='proyekmikon1', password='iee2031',
                              host='10.252.240.62',
                              database='mikon1')


mycursor = mydb.cursor()

mycursor.execute("SHOW DATABASES")

for x in mycursor:
    print(x)

mycursor.execute("SHOW TABLES")
for y in mycursor:
    print(y)

def input_new_value (date,file,case):
    ls = str(date), file,case
    sql = "INSERT INTO DATA_MIKON1_db (DATE, FILE_NAME , status) VALUE (%s, %s, %s)"
    mycursor.execute(sql, ls)
    mydb.commit()
    
def show_all():
    mycursor.execute("SELECT * FROM DATA_MIKON1_db")
    for i in mycursor:
        print(i)
        
def create_new_table():
    mycursor = mydb.cursor()
    state = "CREATE TABLE DATA_MIKON1_db (id INT AUTO_INCREMENT, DATE VARCHAR(150),FILE_NAME VARCHAR(150),status VARCHAR(200), PRIMARY KEY(id))"
    mycursor.execute(state)
    mydb.commit()
    
def delete_table():
    mycursor = mydb.cursor()
    de = "DROP TABLE DATA_MIKON1_db"
    mycursor.execute(de)
    mydb.commit()
    
#delete_table()
create_new_table()
input_new_value("29-01-2022", "halojpg", "no mask")
show_all()