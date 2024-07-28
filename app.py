import streamlit as st
from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai
import pandas as pd
from PyPDF2 import PdfReader

load_dotenv()


def create_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    api_key = "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"

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


def extract_course_codes_rag(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    chain = load_qa_chain(llm=llm, chain_type="stuff")

    query = "Extract and list all course codes from this transcript. Course codes typically consist of 2-4 uppercase letters followed by a space and 3-4 digits, like 'PHIL 1145' or 'CS 5800'. Provide only the list of course codes, separated by commas."

    docs = vectorstore.similarity_search(query=query, k=3)

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)


    return [code.strip() for code in response.split(',')]


def main():
    st.set_page_config(page_title="AI Academic Advisor", page_icon="🎓", layout="wide")

    st.title("AI Academic Advisor")
    st.markdown("Your virtual guide to academic planning!")

    # Sidebar for user inputs
    st.sidebar.header("Student Information")

    university = st.sidebar.selectbox(
        "Select Your University",
        ["Northeastern University", "UC Merced"]
    )

    major = st.sidebar.selectbox(
        "Select Your Major",
        ["Computer Science"]
    )

    # File uploader for transcript
    uploaded_file = st.sidebar.file_uploader("Upload your transcript (PDF)", type="pdf")

    courses_taken = []
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create a vectorstore from the transcript text
        transcript_vectorstore = create_vectorstore(text)

        # Extract course codes using RAG
        courses_taken = extract_course_codes_rag(transcript_vectorstore)

        st.sidebar.success(f"Extracted {len(courses_taken)} course codes from your transcript.")

        # Display extracted courses to the user
        if courses_taken:
            st.sidebar.write("Extracted courses:")
            st.sidebar.write(", ".join(courses_taken))
        else:
            st.sidebar.warning("No course codes were extracted. Please check your transcript format.")

    if university == "Northeastern University":
        csv_filename = "northeastern_university_courses.csv"
    elif university == "UC Merced":
        csv_filename = "uc_merced_courses.csv"
    else:
        st.error("Invalid university selection.")
        return

    csv_file_path = os.path.join("data", csv_filename)

    if not os.path.exists(csv_file_path):
        st.error(f"Course data file for {university} not found.")
        return

    df = pd.read_csv(csv_file_path)
    text = "\n".join(df.values.astype(str).flatten())

    store_name = f"{university.lower().replace(' ', '_')}_{os.path.splitext(os.path.basename(csv_file_path))[0]}"

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        VectorStore = create_vectorstore(text)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

    if st.sidebar.button("Generate Course Plan"):
        if not courses_taken:
            st.warning("Please upload your transcript before generating the course plan...")
            return

        courses_taken_str = ", ".join(courses_taken)

        prompt = f"""Imagine you are an academic advisor for a {major} program at {university}. Your task is to create a detailed, semester-by-semester course schedule for a student majoring in {major}. The student has already taken the following courses: {courses_taken_str}.

Start directly with the first semester's course list. This schedule should span four years (eight semesters) and include all required courses for the major, as well as general education requirements and electives.

When creating this schedule, please consider the following: (if not applicable then create a generals schedule excluding the courses already taken and do not mention this to the user!)
1. Start with the foundational courses in the first year, such as introductory computer science classes, mathematics, and general education requirements.
2. Ensure that prerequisite courses are taken before more advanced courses in the same subject area.
3. Distribute the workload evenly across semesters, aiming for about 15-18 credit hours per semester.
4. Include core computer science courses such as data structures, algorithms, computer systems, and software engineering.
5. Incorporate mathematics courses required for the major, such as calculus and discrete mathematics.
6. Add in general education requirements like writing, social sciences, and humanities courses.
7. Include elective courses in computer science and related fields to allow for specialization.
8. If applicable, factor in any co-op or internship semesters.
9. Ensure that upper-level courses and capstone projects are scheduled in the later years.
10. Include any required courses in related fields, such as electrical engineering or statistics.
11. Consider including courses that fulfill multiple requirements simultaneously when possible.
12. Leave some flexibility in the later semesters for advanced electives or concentration courses.

Please provide a semester-by-semester breakdown, listing the courses for each term along with their credit hours. Also, include a brief explanation for why certain courses are placed in specific semesters.

Remember to balance the technical computer science courses with general education requirements and electives to create a well-rounded academic experience. The schedule should prepare the student for a career in computer science while also providing a broad-based education.

Start directly with the course list for Year 1, Semester 1 without any introductory text."""

        docs = VectorStore.similarity_search(query=prompt, k=3)
        llm = ChatOpenAI(model_name="gpt-4-1106-preview")
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        with st.spinner("Generating your personalized course plan..."):
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=prompt)

        st.subheader("Your Personalized Course Plan")
        st.markdown(response)

        st.sidebar.success("Course plan generated successfully!")


if __name__ == '__main__':
    main()
