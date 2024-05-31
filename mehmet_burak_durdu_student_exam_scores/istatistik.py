import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DataAnalyzer:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
    
    def preprocess_data(self):
        del self.data["Unnamed: 0"]
        self.data.fillna(method="bfill", inplace=True)
    
    def analyze_gender(self):
        gender_counts = np.array([0, 0])
        for i in self.data["Gender"]:
            if i == "male":
                gender_counts[0] += 1
            else:
                gender_counts[1] += 1
        
        labels = ["male", "female"]

        plt.subplot(1, 2, 1)
        plt.bar(labels, gender_counts, color=("#a0ff00", "#ffff00"))
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.title("Distribution of Gender")

        plt.subplot(1, 2, 2)
        colors = ("#a0ff00", "#ffff00")
        plt.pie(gender_counts, labels=labels, colors=colors)
        plt.title("Distribution of Gender")

        plt.tight_layout()
        plt.show()

    def analyze_ethnic_group(self):
        ethnic_group_counts = np.array([0, 0, 0, 0])
        for i in self.data["EthnicGroup"]:
            if i == "group A":
                ethnic_group_counts[0] += 1
            elif i == "group B":
                ethnic_group_counts[1] += 1
            elif i == "group C":
                ethnic_group_counts[2] += 1
            elif i == "group D":
                ethnic_group_counts[3] += 1
        
        labels = ["Group A", "Group B", "Group C", "Group D"]

        plt.subplot(1, 2, 1)
        plt.bar(labels, ethnic_group_counts, color=("#f00ffc", "#f00fb3", "#f00f90", "#f00f80"))
        plt.xlabel("Ethnic Group")
        plt.ylabel("Count")
        plt.title("Distribution of Ethnic Groups")

        plt.subplot(1, 2, 2)
        colors = ("#f00ffc", "#f00fb3", "#f00f90", "#f00f80")
        explode = [0, 0, 0.1, 0]
        plt.pie(ethnic_group_counts, labels=labels, startangle=180, shadow=True, colors=colors, explode=explode)
        plt.title("Distribution of Ethnic Groups")

        plt.tight_layout()
        plt.show()
        print(ethnic_group_counts)

    def analyze_college_education(self):
        college = np.array([0, 0, 0, 0, 0, 0])
        label = ["bachelor's degree", "some college", "master's degree", "associate's degree", "high school", "some high school"]
        educ_place = np.array([0, 0])
        place_label = ["Same University", "Other University"]

        for i in self.data["ParentEduc"]:
            if i == "bachelor's degree":
                college[0] += 1
                educ_place[0] += 1
            if i == "some college":
                college[1] += 1
                educ_place[1] += 1
            if i == "master's degree":
                college[2] += 1
                educ_place[0] += 1
            if i == "associate's degree":
                college[3] += 1
                educ_place[0] += 1
            if i == "high school":
                college[4] += 1
                educ_place[0] += 1
            if i == "some high school":
                college[5] += 1
                educ_place[1] += 1

        college.sort()

        plt.subplot(2, 1, 1)
        plt.barh(label, college, color=("#f00ffc", "#f00fb3", "#f00f90", "#f00f80", "#f00f70", "#f00f60"))
        plt.xlabel("Student Educ Years")
        plt.ylabel("Count")
        plt.title("Parent Education Distribution")

        plt.subplot(2, 1, 2)
        plt.barh(place_label, educ_place, height=0.2, color=("#f00ffc", "#f00f80"))
        plt.xlabel("Count")
        plt.ylabel("Education Place")
        plt.title("Distribution of Education Places")

        plt.tight_layout()
        plt.show()

    def analyze_lunch_type(self):
        lunch_counts = self.data["LunchType"].value_counts()
        
        plt.subplot(1, 2, 1)
        plt.barh(lunch_counts.index, lunch_counts.values, color=["#a0ff00", "#ffff00", "#00aaff"])
        plt.xlabel("Count")
        plt.ylabel("Lunch Type")
        plt.title("Distribution of Lunch Types")

        plt.subplot(1, 2, 2)
        plt.pie(lunch_counts.values, labels=lunch_counts.index, colors=["#a0ff00", "#ffff00", "#00aaff"])
        plt.title("Distribution of Lunch Types")

        plt.tight_layout()
        plt.show()
    
    def analyze_course_completion(self):
        course = np.array([0, 0])
        info_course = ["not completed", "completed"]

        for i in self.data["TestPrep"]:
            if i == "none":
                course[0] += 1
            if i == "completed":
                course[1] += 1

        plt.subplot(2, 1, 1)
        plt.barh(info_course, course, color=("#a0ff00", "#ffff00"))
        plt.xlabel("Count")
        plt.ylabel("Course Completion")
        plt.title("Course Completion Distribution")

        plt.subplot(2, 1, 2)
        plt.pie(course, labels=info_course, colors=("#a0ff00", "#ffff00"))
        plt.title("Course Completion Distribution")

        plt.tight_layout()
        plt.show()

    def analyze_marital_status(self):
        married = np.array([0, 0, 0, 0])
        label = ["married", "single", "widowed", "divorced"]

        for i in self.data["ParentMaritalStatus"]:
            if i == "married":
                married[0] += 1
            if i == "single":
                married[1] += 1
            if i == "widowed":
                married[2] += 1
            if i == "divorced":
                married[3] += 1

        plt.subplot(2, 1, 1)
        plt.bar(label, married, color=("#f00ffc", "#f00fb3", "#f00f90", "#f00f80"))
        plt.xlabel("Marital Status")
        plt.ylabel("Count")
        plt.title("Parent Marital Status Distribution")

        plt.subplot(2, 1, 2)
        plt.pie(married, labels=label, colors=("#f00ffc", "#f00fb3", "#f00f90", "#f00f80"))
        plt.title("Parent Marital Status Distribution")

        plt.tight_layout()
        plt.show()

        print("married:", married[0])
        print("single:", married[1])
        print("widowed:", married[2])
        print("divorced:", married[3])
    
    def analyze_sport(self):
        sport_counts = self.data["PracticeSport"].value_counts()

        plt.subplot(1, 2, 1)
        plt.barh(sport_counts.index, sport_counts.values, color=["#a0ff00", "#ffff00", "#00aaff", "#ff00ff", "#ff0000"])
        plt.xlabel("Count")
        plt.ylabel("Sport")
        plt.title("Distribution of Sports")

        plt.subplot(1, 2, 2)
        plt.pie(sport_counts.values, labels=sport_counts.index, colors=["#a0ff00", "#ffff00", "#00aaff", "#ff00ff", "#ff0000"])
        plt.title("Distribution of Sports")

        plt.tight_layout()
        plt.show()

    def analyze_first_child_status(self):
        firstchild_counts = self.data["IsFirstChild"].value_counts()

        # Çubuk Grafik
        plt.subplot(1, 2, 1)
        plt.barh(firstchild_counts.index, firstchild_counts.values, color=["#a0ff00", "#ffff00"])
        plt.xlabel("Count")
        plt.ylabel("Is First Child")
        plt.title("Distribution of First Child Status")

        # Pasta Grafiği
        plt.subplot(1, 2, 2)
        plt.pie(firstchild_counts.values, labels=firstchild_counts.index, colors=["#a0ff00", "#ffff00"])
        plt.title("Distribution of First Child Status")

        plt.tight_layout()
        plt.show()

    def analyze_siblings(self):
        siblings_counts = self.data["NrSiblings"].value_counts()

        plt.subplot(1, 2, 1)
        plt.barh(siblings_counts.index, siblings_counts.values, color="#a0ff00")
        plt.xlabel("Count")
        plt.ylabel("Number of Siblings")
        plt.title("Distribution of Number of Siblings")

        plt.subplot(1, 2, 2)
        plt.pie(siblings_counts.values, labels=siblings_counts.index, colors=["#a0ff00", "#ffff00", "#00aaff", "#ff00ff"])
        plt.title("Distribution of Number of Siblings")

        plt.tight_layout()
        plt.show()
    
    def analyze_weekly_study_hours(self):
        study_hours_counts = self.data["WklyStudyHours"].value_counts()

        plt.subplot(1, 2, 1)
        plt.barh(study_hours_counts.index, study_hours_counts.values, color="#a0ff00")
        plt.xlabel("Count")
        plt.ylabel("Weekly Study Hours")
        plt.title("Distribution of Weekly Study Hours")

        plt.subplot(1, 2, 2)
        plt.pie(study_hours_counts.values, labels=study_hours_counts.index, colors=["#a0ff00", "#ffff00", "#00aaff", "#ff00ff"])
        plt.title("Distribution of Weekly Study Hours")

        plt.tight_layout()
        plt.show()
    
    def analyze_travel_means(self):
        travel = np.array([0, 0])
        label = ["School Bus", "Private"]

        for i in self.data["TransportMeans"]:
            if i == "school_bus":
                travel[0] += 1
            if i == "private":
                travel[1] += 1

        plt.subplot(2, 1, 1)
        plt.barh(label, travel, color=("#a0ff00", "#ffff00"))
        plt.subplot(2, 1, 2)
        plt.pie(travel, labels=label, colors=("#a0ff00", "#ffff00"))
        plt.show()

        print("School Bus travel:", travel[0])
        print("Private vehicle travel:", travel[1])
        
    def analyze_math_scores(self):
        math_scores = self.data["MathScore"]

        plt.subplot(1, 2, 1)
        plt.hist(math_scores, bins=10, color="#a0ff00")
        plt.xlabel("Math Score")
        plt.ylabel("Count")
        plt.title("Distribution of Math Scores")

        plt.subplot(1, 2, 2)
        plt.pie(math_scores.value_counts(), labels=math_scores.value_counts().index, autopct="%1.1f%%", colors=["#a0ff00", "#ffff00", "#00aaff", "#ff00ff"])
        plt.title("Distribution of Math Scores")

        plt.tight_layout()
        plt.show()
    
    def analyze_reading_scores(self):
        reading_scores = self.data["ReadingScore"]

        plt.subplot(1, 2, 1)
        plt.hist(reading_scores, bins=10, color="#a0ff00")
        plt.xlabel("Reading Score")
        plt.ylabel("Count")
        plt.title("Distribution of Reading Scores")

        plt.subplot(1, 2, 2)
        plt.pie(reading_scores.value_counts(), labels=reading_scores.value_counts().index, autopct="%1.1f%%", colors=["#a0ff00", "#ffff00", "#00aaff", "#ff00ff"])
        plt.title("Distribution of Reading Scores")

        plt.tight_layout()
        plt.show()
    
    def analyze_writing_scores(self):
        writing_scores = self.data["WritingScore"]
        plt.subplot(1, 2, 1)
        plt.hist(writing_scores, bins=10, color="#a0ff00")
        plt.xlabel("Writing Score")
        plt.ylabel("Count")
        plt.title("Distribution of Writing Scores")

        plt.show()
    
    def plot_summary_statistics(self):
        summary_stats = self.data.describe()
        print(summary_stats)

    

    def run_analysis(self):
        self.preprocess_data()

        self.analyze_gender()
        self.analyze_ethnic_group()
        self.analyze_college_education()
        self.analyze_lunch_type()
        self.analyze_course_completion()
        self.analyze_marital_status()
        self.analyze_sport()
        self.analyze_first_child_status()
        self.analyze_siblings()
        self.analyze_travel_means()
        self.analyze_weekly_study_hours()
        self.analyze_math_scores()
        self.analyze_reading_scores()
        self.analyze_writing_scores()
        self.plot_summary_statistics()

# Create an instance of DataAnalyzer and run the analysis
analyzer = DataAnalyzer("Expanded_data_with_more_features.csv")
analyzer.run_analysis()
