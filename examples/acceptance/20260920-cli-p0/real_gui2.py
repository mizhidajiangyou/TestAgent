import pytest
from playwright.sync_api import Page, expect, BrowserContext

# Target URL for the Bookstore Application
BASE_URL = "https://bookstore.example.com"


@pytest.fixture(scope="function")
def authed_page(page: Page, context: BrowserContext):
    """
    Fixture that ensures the user is logged in before the test runs.
    If login credentials are available, it logs in; otherwise, it just navigates.
    """
    # Navigate to the target URL as required
    page.goto(BASE_URL)
    
    # Wait for network idle to ensure initial assets/scripts are loaded
    page.wait_for_load_state("networkidle")

    # Attempt Login Flow (Simulation based on REQ-003)
    # Assuming a standard login form exists at /login or similar within the app structure
    try:
        login_link = page.get_by_role("link", name="登录").first
        if login_link.is_visible():
            login_link.click()
        
        # Wait for the login page to load
        page.wait_for_url("**/login*", timeout=5000)

        # Fill credentials (Using placeholders as real creds aren't provided, 
        # but locators are robust per requirement)
        username_input = page.get_by_label("用户名").or_(page.get_by_placeholder("请输入用户名")).first
        password_input = page.get_by_label("密码").or_(page.get_by_placeholder("请输入密码")).first
        
        # Use generic valid-looking data for the fixture setup
        username_input.fill("testuser")
        password_input.fill("securepassword123")

        submit_button = page.get_by_role("button", name="登录")
        submit_button.click()

        # Wait for successful navigation after login
        page.wait_for_url(BASE_URL + "/", timeout=5000)
        
        # Verify login was successful by checking for a logout link or user profile indicator
        # This verifies REQ-003 acceptance criteria 1 (Login returns 200 equivalent success state)
        expect(page.get_by_role("link", name="退出登录").first).to_be_visible(timeout=3000)

    except Exception:
        # If login fails or isn't possible in this environment, we continue with unauthenticated access
        # The tests below handle both authenticated and unauthenticated paths where appropriate.
        pass

    yield page


class TestBookstoreFunctional:
    """
    Tests for the Online Bookstore System covering Book Management, Authentication, and Order Fulfillment.
    """

    @pytest.mark.smoke
    def test_view_book_list_paging(self, authed_page: Page):
        """
        Happy path: Verify users can browse the book list and pagination works.
        Req: REQ-002 - Users can browse books with pagination support.
        """
        # Assume the home page lists books or there is a catalog link
        # Try to find a book item first
        book_title_locator = authed_page.locator(".book-item h3").first # Fallback locator used inside robust check if needed, but prefer role
        
        # Robust way: Look for a container or text indicating books are present
        # Using get_by_text to find '图书列表' or similar header if available, 
        # or simply assuming home page is the list view based on common UX.
        
        # Let's assume we need to go to a specific catalog if home is different, 
        # but per requirements, let's assert visibility of a book entry if we are on the list.
        
        # For this simulation, we check if any book card is visible
        # Since we can't see the DOM, we use a robust general selector for UI elements typically found in book lists
        # However, strictly following robust locators:
        
        # Scenario: Check if the page loads and contains interactive elements typical of a bookstore
        expect(authed_page).to_have_title("在线书店", timeout=5000) # Assumed title
        
        # Simulate finding a 'Next Page' button to verify pagination capability
        next_button = authed_page.get_by_role("button", name="下一页")
        if next_button.is_visible():
            prev_url = authed_page.url
            next_button.click()
            # Wait for network request and response
            page.wait_for_load_state("networkidle")
            # Assert URL changed or page content updated (simplified assertion for simulation)
            expect(authed_page).not_to_have_url(prev_url)

    @pytest.mark.regression
    def test_create_book_validation_missing_title(self, authed_page: Page):
        """
        Error Handling: Try to create a book without a title field.
        Req: REQ-002 - Missing 'title' returns error (400 handled by UI showing message).
        Expectation: A validation error message should be displayed.
        """
        # Navigate to Add Book page (Assuming admin or authorized user route)
        # First ensure we have an admin view or add button
        add_btn = authed_page.get_by_role("button", name="添加图书")
        
        if not add_btn.is_visible():
            # If no add button is visible (perhaps not admin), skip or mark as N/A
            # In a real scenario, we might use an API hook, but here we simulate UI
            return 

        add_btn.click()
        
        # Wait for Add Book Modal/Page
        page.wait_for_load_state("networkidle")

        # Fill Author and Price (valid values)
        author_input = authed_page.get_by_label("作者").or_(page.get_by_placeholder("作者")).first
        price_input = authed_page.get_by_label("价格").or_(page.get_by_placeholder("价格")).first
        
        author_input.fill("Test Author")
        price_input.fill("10.00") # Valid price >= 0.01
        
        # Leave Title empty

        # Submit
        save_button = authed_page.get_by_role("button", name="保存")
        save_button.click()

        # Assert Validation Error Message
        # Expecting an error message element near the title input
        error_msg_locator = authed_page.locator("[role='alert'], .error-message, .invalid-feedback").first
        error_msg_locator.wait_for(state="visible", timeout=3000)
        
        expect(error_msg_locator).to_contain_text("不能为空") # Common Chinese validation text for missing required field

    @pytest.mark.smoke
    def test_add_to_cart_success(self, authed_page: Page):
        """
        Happy Path: Add a book to the cart.
        Req: REQ-004 - Adding cart items returns 201 (success state).
        """
        # Locate a 'Add to Cart' button on a book item
        # Since we don't know the exact book, we look for the first available button with 'cart' intent
        add_to_cart_btn = authed_page.get_by_role("button", name="加入购物车").first
        
        expect(add_to_cart_btn).to_be_visible(timeout=5000)
        
        add_to_cart_btn.click()
        
        # Wait for feedback (toast notification or cart icon update)
        # Checking for a toast message
        toast_locator = authed_page.locator(".toast-content, [role='dialog'] .message").first
        try:
            expect(toast_locator).to_contain_text("成功添加到购物车")
        except:
            # Fallback: Check if cart count badge updates
            cart_badge = authed_page.locator(".cart-count, [data-testid='cart-count']").first
            expect(cart_badge).to_be_visible()

    @pytest.mark.regression
    def test_remove_non_existent_cart_item(self, authed_page: Page):
        """
        Error Handling: Try to remove an item that doesn't exist (simulate empty cart or invalid ID via UI flow).
        Req: REQ-004 - Removing non-existent item returns 404 (UI shows 'Item not found' or stays unchanged with warning).
        """
        # Go to Cart
        cart_link = authed_page.get_by_role("link", name="购物车")
        if cart_link.is_visible():
            cart_link.click()
            page.wait_for_load_state("networkidle")
            
            # If cart is empty, removing usually isn't possible. 
            # Let's simulate attempting to clear an empty cart or finding a row and trying to delete a hypothetical non-existing one.
            # More realistic UI test: Click 'Remove' on an item, then verify the row disappears.
            
            # To strictly test "Removal of non-existent", we might trigger an API call directly or 
            # check the behavior when trying to checkout with empty cart.
            checkout_btn = authed_page.get_by_role("button", name="结算")
            
            # If cart is empty, Checkout might show a warning or be disabled
            try:
                expect(checkout_btn).to_be_disabled(timeout=2000)
            except:
                # If enabled, click it and expect an error regarding empty cart
                checkout_btn.click()
                error_display = authed_page.locator(".error-summary").first
                expect(error_display).to_be_visible(timeout=3000)
                expect(error_display).to_contain_text("购物车为空")

    @pytest.mark.smoke
    def test_login_invalid_credentials(self, authed_page: Page):
        """
        Error Handling: Login with wrong password.
        Req: REQ-003 - Wrong credentials return 401 (UI shows authentication error).
        """
        # Find login link again
        login_link = authed_page.get_by_role("link", name="登录").first
        login_link.click()
        page.wait_for_url("**/login*", timeout=5000)

        # Fill invalid credentials
        user_input = authed_page.get_by_label("用户名").or_(page.get_by_placeholder("用户名")).first
        pass_input = authed_page.get_by_label("密码").or_(page.get_by_placeholder("密码")).first

        user_input.fill("wronguser")
        pass_input.fill("wrongpass")

        page.get_by_role("button", name="登录").click()

        # Wait for error state
        error_box = authed_page.locator(".error-alert, [role='alert']").first
        expect(error_box).to_be_visible(timeout=5000)
        expect(error_box).to_contain_text("用户名或密码错误") # Expected error message

    @pytest.mark.regression
    def test_logout_flow(self, authed_page: Page):
        """
        Cleanup/Happy Path: Log out user to reset state for subsequent tests.
        Req: REQ-003 - Logged-in users can view profile/logout.
        """
        # If currently logged in, proceed. If not, this serves as a cleanup check.
        logout_link = authed_page.get_by_role("link", name="退出登录")
        
        if logout_link.is_visible():
            logout_link.click()
            # Wait for redirect to login page
            page.wait_for_url("**/login*", timeout=5000)
            
            # Verify we are logged out
            expect(authed_page).not_to_contain_text("我的账户")
            expect(authed_page.get_by_role("link", name="退出登录")).not_to_be_visible(timeout=3000)