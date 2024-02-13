import streamlit as st
def main():
  st.set_page_config(
      page_title = "DirChat",
      page_icon = ":books:"
  )
  st.title("_Private Data : red[QA chat]_:books:")
  print("Hello World")
  st.write("""
  # My first app
  Hello * world! *
  """)

if __name__ == "__main__":
  main()