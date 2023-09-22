document.querySelector('.chat-button').addEventListener('click', function(event) {
    // Prevent the default behavior (following the link)
    event.preventDefault();
    
    // Get the link from the button's href attribute
    const chatPageLink = this.getAttribute('href');
    
    // Redirect to the chat.html page
    window.location.href = chatPageLink;
});