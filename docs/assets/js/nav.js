// Keyboard accessibility for the mobile nav toggle label/checkbox pattern.
// Allows Enter and Space to activate the nav toggle when focused via keyboard.
(function () {
  var label = document.querySelector('label[for="nav-toggle"]');
  if (!label) return;
  label.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      label.click();
    }
  });
}());
