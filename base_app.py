"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# For showing progress while computing predictions
import time

# Import libraries for visuals
import matplotlib.pyplot as plt
import seaborn as sns

# Import  libraries for NLP
import nltk
from nltk import pos_tag as pos
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

# Vectorizer
vectorizer = open("resources/pipeline.pickle","rb")
tweet_cv = joblib.load(vectorizer) # loading your vectorizer from the pkl file

# Create an object of class PorterStemmer and WordNetLemmatizer
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load your raw data
raw = pd.read_csv("resources/train.csv")		

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.markdown("<h1 style='color: orange;text-align:center;'>Climate Change Belief Analysis</h1>", unsafe_allow_html=True)
	st.text("Predict an individual’s belief in climate change based on historical tweet data")

	# Set image names
	words_jpg = ['resources/all_words0.jpg', 'resources/news0.jpg', 'resources/pro0.jpg', 'resources/neutral0.jpg', 'resources/anti0.jpg']
	cloud_jpg = ['resources/all_words1.jpg', 'resources/news1.jpg', 'resources/pro1.jpg', 'resources/neutral1.jpg', 'resources/anti1.jpg']
	handl_jpg = ['resources/news2.jpg', 'resources/pro2.jpg', 'resources/neutral2.jpg', 'resources/anti2.jpg']
	hasht_jpg = ['resources/news3.jpg', 'resources/pro3.jpg', 'resources/neutral3.jpg', 'resources/anti3.jpg']

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "EDA", "NLP","Make Predictions", "The Team"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		page = """	
				* Home
				* Read about the project
				* Use 'choose option' to change page
				"""
		st.sidebar.markdown(page)
		st.markdown('<h4 style="border-radius: 5px;background: orange;padding: 7px; width: 400px;opacity: 90%",>General Information</h4>', unsafe_allow_html=True)
		# You can read a markdown file from supporting resources folder
		st.markdown("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat")

		# Problem Statement
		st.markdown('<h4 style="border-radius: 5px;background: orange;padding: 7px; width: 400px;opacity: 90%",>Problem Statement</h4>', unsafe_allow_html=True)
		st.markdown("Build a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data.")
        
		# Values Proposition
		st.markdown('<h4 style="border-radius: 5px;background: orange;padding: 7px; width: 400px;opacity: 90%",>Value Proposition</h4>', unsafe_allow_html=True)
		st.markdown("This would add to their market research efforts in gauging how their product/service may be received. Providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies.")


		st.subheader("Raw Twitter data and label")
		st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following classes:")
        
		# Classes
		st.markdown('<h4 style="border-radius: 5px;background: orange;padding: 7px; width: 400px;opacity: 90%",>Classes</h4>', unsafe_allow_html=True)
		st.markdown("sentiment:")

		# Class description
		class_desc = """| Name | Description         
						| :-: | :-------------
						| 2 | **News:** the tweet link to factual news about climate change
						| 1 | **Pro:** the tweet supports the belief of man-made climate change
						| 0 | **Neutral:** the tweet neither supports nor refutes the belief of man-made climate change
						| -1 | **Anti:** the tweet does not believe in man-made climate change
					"""
		st.markdown(class_desc)
                   
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "EDA" page
	if selection == "EDA":
		st.markdown("<h2 style='color: black;text-align:center;'>Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
		# Creating sidebar with selection box -
		# you can create multiple pages this way
		page = """	
				* EDA
				* Use the checkbox to view the visualization
				* Use 'choose option' to change page
				"""
		st.sidebar.markdown(page)
		plot_options = ["Most Common Words", "Word cloud", "Twitter Handles","Hashtags"]
		select = st.sidebar.selectbox("Visuals", plot_options)

		if select == "Most Common Words":
			st.markdown('<h3 style="border-radius: 5px;background: orange;padding: 7px; width: 400px;opacity: 90%",>Use the checkboxes below to explore the most popular word in the tweets</h3>', unsafe_allow_html=True)
			if st.checkbox("All Tweets"):
				st.image(words_jpg[0])
			if st.checkbox("Pro Tweets"):
				st.image(words_jpg[1])
			if st.checkbox("Neutral Tweets"):
				st.image(words_jpg[2])
			if st.checkbox("Anti Tweets"):
				st.image(words_jpg[3])
			if st.checkbox("News Tweets"):
				st.image(words_jpg[4])

		if select == "Word cloud":
			st.markdown('<h3 style="border-radius: 5px;background: orange;padding: 7px; width: 400px;opacity: 90%",>Use the checkboxes below to explore a Word Cloud of the tweets</h3>', unsafe_allow_html=True)
			if st.checkbox("All Tweets"):
				st.image(cloud_jpg[0])
			if st.checkbox("Pro Tweets"):
				st.image(cloud_jpg[1])
			if st.checkbox("Neutral Tweets"):
				st.image(cloud_jpg[2])
			if st.checkbox("Anti Tweets"):
				st.image(cloud_jpg[3])
			if st.checkbox("News Tweets"):
				st.image(cloud_jpg[4])

		if select == "Twitter Handles":
			st.markdown('<h3 style="border-radius: 5px;background: orange;padding: 7px; width: 400px;opacity: 90%",>Use the checkboxes below to explore the most popular handles in the tweets</h3>', unsafe_allow_html=True)
			if st.checkbox("Pro Tweets"):
				st.image(handl_jpg[0])
			if st.checkbox("Neutral Tweets"):
				st.image(handl_jpg[1])
			if st.checkbox("Anti Tweets"):
				st.image(handl_jpg[2])
			if st.checkbox("News Tweets"):
				st.image(handl_jpg[3])

		if select == "Hashtags":
			st.markdown('<h3 style="border-radius: 5px;background: orange;padding: 7px; width: 400px;opacity: 90%",>Use the checkboxes below to explore the most popular hashtags in the tweets</h3>', unsafe_allow_html=True)
			if st.checkbox("Pro Tweets"):
				st.image(hasht_jpg[0])
			if st.checkbox("Neutral Tweets"):
				st.image(hasht_jpg[1])
			if st.checkbox("Anti Tweets"):
				st.image(hasht_jpg[2])
			if st.checkbox("News Tweets"):
				st.image(hasht_jpg[3])

	# Building "NLP" page
	if selection == 'NLP':
		page = """	
				* NLP (Natural Languge Processesing)
				* Perform NLP tasks
				* Use 'choose option' to change page
				"""
		st.sidebar.markdown(page)
		st.markdown("<h2 style='color: black;text-align:center;'>Natural Language Processesing (NLP)</h2>", unsafe_allow_html=True)
		raw_text = st.text_area("Enter News Here","Type Here")
		nlp_task = ["Tokenization", "Stemming", "Lemmatization","NER","POS Tags", 'Sentiment score']
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("Original Text::\n{}".format(raw_text))
			tokens = raw_text.split(" ")
			df = pd.DataFrame()
			if task_choice == 'Tokenization':
				indices = [i for i in tokens]
				df['Token'] = tokens
				df['Index'] = indices
				st.write(df)
			elif task_choice == 'Stemming':
				result = [porter.stem(x) for x in tokens]
				df['Token'] = tokens
				df['Stem'] = result
				st.write(df)
			elif task_choice == 'Lemmatization':
				result = [lemmatizer.lemmatize(x) for x in tokens]
				df['Token'] = tokens
				df['Lemma'] = result
				st.write(df)
			elif task_choice == 'POS Tags':
				result = pos(result, tagset='universal')
				df = pd.DataFrame(result, columns=['Token', 'POS Tag'])
				st.write(df)
			elif task_choice == 'Sentiment score':
				p = TextBlob(raw_text).sentiment.polarity
				s = TextBlob(raw_text).sentiment.subjectivity
				st.markdown("""
							* Polarity is a score ranging from -1 to 1, it describes the general sentiment of a text. 1:positive, 0:neutral and -1:negative
							* Subjectivity is a score ranging from 0 to 1, it tell us whether the text is subjective or objective. 0:objective and 1:subjective
							""")
				st.markdown('Polarity: {}\nSubjectivity: {}'.format(p, s))

# Building out the predication page.
	if selection == "Make Predictions":
			""" Please Choose a Model of your choice"""
			st.markdown("<h2 style='color: black;text-align:center;'>Make Prediction</h2>", unsafe_allow_html=True)
			page = """	
				* Make Predictions
				* Enter sample tweet and pick a model to classify the tweet
				* Use 'choose option' to change page
				"""
			st.sidebar.markdown(page)
			options = ["Logistics Regression", "Linear SVC", "Original"]
			selection = st.selectbox("Choose Model", options)
# 			st.info(selection)
			""" Please enter the tweet you would like to make predictions on."""
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Tweet","Type Here")
			""" You are about to classify the tweet using""" 
			st.info(selection)
			# Now, let’s create a progress bar:
			# Add a placeholder
			""" computing... please wait""" 
# 			latest_iteration = st.empty()
			bar = st.progress(0) 

			for i in range(100):
				# Update the progress bar with each iteration.
# 				latest_iteration.text(f'Iteration {i+1}')
				bar.progress(i + 1)   
				time.sleep(0.1) 

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text])
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice.
				if selection == "Logistics Regression":
					predictor = joblib.load(open(os.path.join("resources/log_reg_model.pickle"),"rb"))
					prediction = predictor.predict(vect_text)
				if selection == "Linear SVC":
					predictor = joblib.load(open(os.path.join("resources/lin_svc_model.pickle"),"rb"))
					prediction = predictor.predict(vect_text)
				if selection == "Original":
					predictor = joblib.load(open(os.path.join("resources/logistic_regression.pkl"),"rb"))
					prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

	# Building out the team page.
	if selection == "The Team":
		page = """	
			* The Team
			* Use 'choose option' to change page
			"""
		st.sidebar.markdown(page)
		cl = """
				<div style="margin-top: 50px;">
					<h1 style="font-size: 52px;margin-bottom: 60px;text-align: center;">Meet the Team</h1>
					<div style="  display: flex;justify-content: center;width: auto;text-align: center;flex-wrap: wrap;">
						<div style="background: #f0f2f6;border-radius: 5%;margin: 5px;margin-bottom: 50px;width: 300px;padding: 20px;line-height: 20px;color: #8e8b8b;position: relative;">
						<div style="position: absolute;top: -50px;left: 50%;transform: translateX(-50%);width: 100px;height: 100px;border-radius: 50%;background: #FFA500;">
							<img src="https://athena.explore-datascience.net/uploads/profile/3668-wryj.jpg" alt="Team_image" style="width: 100px;height: 100px;padding: 5px;border-radius: 50%">
						</div>
						<h3 style="color: black;font-family: "Comic Sans MS", cursive, sans-serif;font-size: 26px;margin-top: 50px;">Lizwi Khanyile</h3>
						<p style="color: #262730;margin: 12px 0;font-size: 17px;text-transform: uppercase;">ML Engineer</p>
						<p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Est quaerat tempora, voluptatum quas facere dolorum aut cumque nihil nulla harum nemo distinctio quam blanditiis dignissimos.</p>
						<ul>
  							<li style="display:inline;">
								<a href="#"><img border="0" alt="Twitter" src="https://image.flaticon.com/icons/svg/1384/1384017.svg" width="25" height="25"></a>  
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Linkein" src="https://image.flaticon.com/icons/svg/1384/1384014.svg" width="25" height="25"></a>
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Github" src="https://image.flaticon.com/icons/svg/25/25231.svg" width="25" height="25"></a>
							</li>
						</ul>
						</div>
												<div style="background: #f0f2f6;border-radius: 5%;margin: 5px;margin-bottom: 50px;width: 300px;padding: 20px;line-height: 20px;color: #8e8b8b;position: relative;">
						<div style="position: absolute;top: -50px;left: 50%;transform: translateX(-50%);width: 100px;height: 100px;border-radius: 50%;background: #FFA500;">
							<img src="https://athena.explore-datascience.net/uploads/profile/3668-wryj.jpg" alt="Team_image" style="width: 100px;height: 100px;padding: 5px;border-radius: 50%">
						</div>
						<h3 style="color: black;font-family: "Comic Sans MS", cursive, sans-serif;font-size: 26px;margin-top: 50px;">Lizwi Khanyile</h3>
						<p style="color: #262730;margin: 12px 0;font-size: 17px;text-transform: uppercase;">ML Engineer</p>
						<p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Est quaerat tempora, voluptatum quas facere dolorum aut cumque nihil nulla harum nemo distinctio quam blanditiis dignissimos.</p>
						<ul>
  							<li style="display:inline;">
								<a href="#"><img border="0" alt="Twitter" src="https://image.flaticon.com/icons/svg/1384/1384017.svg" width="25" height="25"></a>  
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Linkein" src="https://image.flaticon.com/icons/svg/1384/1384014.svg" width="25" height="25"></a>
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Github" src="https://image.flaticon.com/icons/svg/25/25231.svg" width="25" height="25"></a>
							</li>
						</ul>
						</div>
												<div style="background: #f0f2f6;border-radius: 5%;margin: 5px;margin-bottom: 50px;width: 300px;padding: 20px;line-height: 20px;color: #8e8b8b;position: relative;">
						<div style="position: absolute;top: -50px;left: 50%;transform: translateX(-50%);width: 100px;height: 100px;border-radius: 50%;background: #FFA500;">
							<img src="https://athena.explore-datascience.net/uploads/profile/3668-wryj.jpg" alt="Team_image" style="width: 100px;height: 100px;padding: 5px;border-radius: 50%">
						</div>
						<h3 style="color: black;font-family: "Comic Sans MS", cursive, sans-serif;font-size: 26px;margin-top: 50px;">Lizwi Khanyile</h3>
						<p style="color: #262730;margin: 12px 0;font-size: 17px;text-transform: uppercase;">ML Engineer</p>
						<p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Est quaerat tempora, voluptatum quas facere dolorum aut cumque nihil nulla harum nemo distinctio quam blanditiis dignissimos.</p>
						<ul>
  							<li style="display:inline;">
								<a href="#"><img border="0" alt="Twitter" src="https://image.flaticon.com/icons/svg/1384/1384017.svg" width="25" height="25"></a>  
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Linkein" src="https://image.flaticon.com/icons/svg/1384/1384014.svg" width="25" height="25"></a>
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Github" src="https://image.flaticon.com/icons/svg/25/25231.svg" width="25" height="25"></a>
							</li>
						</ul>
						</div>
												<div style="background: #f0f2f6;border-radius: 5%;margin: 5px;margin-bottom: 50px;width: 300px;padding: 20px;line-height: 20px;color: #8e8b8b;position: relative;">
						<div style="position: absolute;top: -50px;left: 50%;transform: translateX(-50%);width: 100px;height: 100px;border-radius: 50%;background: #FFA500;">
							<img src="https://athena.explore-datascience.net/uploads/profile/3668-wryj.jpg" alt="Team_image" style="width: 100px;height: 100px;padding: 5px;border-radius: 50%">
						</div>
						<h3 style="color: black;font-family: "Comic Sans MS", cursive, sans-serif;font-size: 26px;margin-top: 50px;">Lizwi Khanyile</h3>
						<p style="color: #262730;margin: 12px 0;font-size: 17px;text-transform: uppercase;">ML Engineer</p>
						<p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Est quaerat tempora, voluptatum quas facere dolorum aut cumque nihil nulla harum nemo distinctio quam blanditiis dignissimos.</p>
						<ul>
  							<li style="display:inline;">
								<a href="#"><img border="0" alt="Twitter" src="https://image.flaticon.com/icons/svg/1384/1384017.svg" width="25" height="25"></a>  
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Linkein" src="https://image.flaticon.com/icons/svg/1384/1384014.svg" width="25" height="25"></a>
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Github" src="https://image.flaticon.com/icons/svg/25/25231.svg" width="25" height="25"></a>
							</li>
						</ul>
						</div>
												<div style="background: #f0f2f6;border-radius: 5%;margin: 5px;margin-bottom: 50px;width: 300px;padding: 20px;line-height: 20px;color: #8e8b8b;position: relative;">
						<div style="position: absolute;top: -50px;left: 50%;transform: translateX(-50%);width: 100px;height: 100px;border-radius: 50%;background: #FFA500;">
							<img src="https://athena.explore-datascience.net/uploads/profile/3668-wryj.jpg" alt="Team_image" style="width: 100px;height: 100px;padding: 5px;border-radius: 50%">
						</div>
						<h3 style="color: black;font-family: "Comic Sans MS", cursive, sans-serif;font-size: 26px;margin-top: 50px;">Lizwi Khanyile</h3>
						<p style="color: #262730;margin: 12px 0;font-size: 17px;text-transform: uppercase;">ML Engineer</p>
						<p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Est quaerat tempora, voluptatum quas facere dolorum aut cumque nihil nulla harum nemo distinctio quam blanditiis dignissimos.</p>
						<ul>
  							<li style="display:inline;">
								<a href="#"><img border="0" alt="Twitter" src="https://image.flaticon.com/icons/svg/1384/1384017.svg" width="25" height="25"></a>  
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Linkein" src="https://image.flaticon.com/icons/svg/1384/1384014.svg" width="25" height="25"></a>
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Github" src="https://image.flaticon.com/icons/svg/25/25231.svg" width="25" height="25"></a>
							</li>
						</ul>
						</div>
												<div style="background: #f0f2f6;border-radius: 5%;margin: 5px;margin-bottom: 50px;width: 300px;padding: 20px;line-height: 20px;color: #8e8b8b;position: relative;">
						<div style="position: absolute;top: -50px;left: 50%;transform: translateX(-50%);width: 100px;height: 100px;border-radius: 50%;background: #FFA500;">
							<img src="https://athena.explore-datascience.net/uploads/profile/3668-wryj.jpg" alt="Team_image" style="width: 100px;height: 100px;padding: 5px;border-radius: 50%">
						</div>
						<h3 style="color: black;font-family: "Comic Sans MS", cursive, sans-serif;font-size: 26px;margin-top: 50px;">Lizwi Khanyile</h3>
						<p style="color: #262730;margin: 12px 0;font-size: 17px;text-transform: uppercase;">ML Engineer</p>
						<p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Est quaerat tempora, voluptatum quas facere dolorum aut cumque nihil nulla harum nemo distinctio quam blanditiis dignissimos.</p>
						<ul>
  							<li style="display:inline;">
								<a href="#"><img border="0" alt="Twitter" src="https://image.flaticon.com/icons/svg/1384/1384017.svg" width="25" height="25"></a>  
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Linkein" src="https://image.flaticon.com/icons/svg/1384/1384014.svg" width="25" height="25"></a>
							</li>
  							<li style="display:inline;">
							  	<a href="#"><img border="0" alt="Github" src="https://image.flaticon.com/icons/svg/25/25231.svg" width="25" height="25"></a>
							</li>
						</ul>
						</div>
					</div>
				</div>
			"""
		st.markdown(cl, unsafe_allow_html=True)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
