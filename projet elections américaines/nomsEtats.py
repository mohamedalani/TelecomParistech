f = open("C:\\Users\\Mohamed\\MS BGD\\INF728 NoSQL\\projet elections américaines\\uStates.js", 'r')

data = f.read()
rows = data.split("\n")
rows = rows[2:]

rows = [row.split(",d:")[0] for row in rows][:51]

dico = ""
for row in rows :
    dico += row.split(",n:")[1]
    dico += ":"
    dico += row.split(",n:")[0].replace("{id:", "")
    dico += ", "
print(dico)
