import sqlite3

conn = sqlite3.connect('projects.db')
c = conn.cursor()

# Table banana (agar nahi hai)
c.execute('''CREATE TABLE IF NOT EXISTS projects
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              title TEXT,
              abstract TEXT)''')

# Sample data
projects = [
    ("AI based Crop Detection", "A project that uses AI to detect crop types and suggest fertilizers."),
    ("Water Leakage Detection using IoT", "Sensors detect leaks in pipelines and notify users via mobile app."),
    ("Smart Traffic System", "Using image processing and ML to manage urban traffic flow efficiently.")
]

# Insert karo
c.executemany("INSERT INTO projects (title, abstract) VALUES (?, ?)", projects)
conn.commit()
conn.close()

print("Seed data added successfully ✅")
