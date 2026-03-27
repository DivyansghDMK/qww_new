class SignOut:
    def sign_out_user(self, main_window=None):
        # Logic for signing out the user
        if main_window:
            main_window.close()
        print("User signed out successfully.")
        return True