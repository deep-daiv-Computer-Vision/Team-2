document.addEventListener("DOMContentLoaded", function() {
    const spans = document.querySelectorAll("span.hoverable");
    spans.forEach((span, index) => {
        span.addEventListener("mouseenter", () => {
            // Hover된 문장의 인덱스를 Streamlit으로 전달
            Streamlit.setComponentValue(index);
        });
    });
});