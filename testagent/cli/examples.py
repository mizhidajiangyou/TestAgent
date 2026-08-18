"""Pre-formatted example/help blocks appended to CLI command help text.

Kept in one place so command modules stay focused on behavior. Each block is
rendered verbatim (no re-wrapping) via :class:`_ExamplesHelpMixin` in
:mod:`testagent.cli.base`.
"""

EXAMPLES_TEXT = """
EXAMPLES:
  # Show current configuration
  testagent config

  # Generate test cases (JSON, default)
  testagent generate-tests -s swagger.json -r requirements.md -o ./output/testcases.json

  # Generate Markdown report
  testagent generate-tests -s swagger.json -r requirements.md -f markdown -o ./output/testcases.md

  # Generate CSV (Excel-friendly, UTF-8 BOM)
  testagent generate-tests -s swagger.json -r requirements.md -f csv -o ./output/testcases.csv

  # Incremental generation: reuse previous cases as baseline, only add net-new cases
  testagent generate-tests -s swagger.json -r new_requirements.md -H ./output/testcases.json

  # Parse Swagger from URL
  testagent generate-tests -s https://petstore3.swagger.io/api/v3/openapi.json

  # Enable multi-model fallback + cross-validation review
  OPENAI_MODEL="gpt-4o-mini,gpt-4o" REVIEW_ENABLED=true REVIEW_MAX_ROUNDS=2 \\
      testagent generate-tests -s swagger.json -r requirements.md -f markdown

  # Generate k6 performance script (default format)
  testagent generate-perf -s swagger.json --base-url https://api.example.com

  # Generate JMeter JMX script
  testagent generate-perf -s swagger.json -f jmeter -o ./output/perf.jmx

  # Generate Playwright GUI test script
  testagent generate-gui -r requirements.md --url https://example.com -o ./output/gui_test.py

  # Interactive conversational refinement (generate then refine via dialogue)
  testagent chat -r requirements.md -s swagger.json

  # Start the web GUI (embeddable via iframe in other platforms)
  testagent serve --port 8000

  # Run generated scripts
  k6 run output/perf_test.js
  jmeter -n -t output/perf_test.jmx -l results.jtl
  pytest output/gui_test.py --browser chromium

ENVIRONMENT:
  All settings can be overridden via env vars or .env file (see .env.example).
  Priority: env vars > .env > defaults.

  OPENAI_MODEL          comma-separated model list, first is primary (default: gpt-4o-mini)
  OPENAI_API_KEY        OpenAI API key
  OPENAI_BASE_URL       OpenAI-compatible endpoint (default: https://api.openai.com/v1)
  REVIEW_ENABLED        run cross-validation review after generation (default: false)
  REVIEW_MAX_ROUNDS     review rounds, odd=secondary model, even=primary (default: 2)
  OUTPUT_LANGUAGE       chinese | english (default: chinese)
  OUTPUT_DIR            output directory (default: ./output)
  SCRIPT_FORMAT         k6 | jmeter (default: k6)
"""

GENERATE_TESTS_EXAMPLES = """
EXAMPLES:
  # JSON output (default)
  testagent generate-tests -s swagger.json -r requirements.md

  # Markdown report
  testagent generate-tests -s swagger.json -r requirements.md -f markdown -o report.md

  # CSV for Excel
  testagent generate-tests -s swagger.json -f csv -o cases.csv

  # Swagger from URL
  testagent generate-tests -s https://petstore3.swagger.io/api/v3/openapi.json

  # With multi-model cross-validation review
  OPENAI_MODEL="gpt-4o-mini,gpt-4o" REVIEW_ENABLED=true \\
      testagent generate-tests -s swagger.json -r requirements.md

  # Requirements only (no Swagger)
  testagent generate-tests -r requirements.md -o cases.json

  # Incremental: reuse a previous test case baseline, only generate net-new cases
  testagent generate-tests -s swagger.json -r new_requirements.md -H ./output/testcases.json
"""

GENERATE_PERF_EXAMPLES = """
EXAMPLES:
  # k6 script (default format from config)
  testagent generate-perf -s swagger.json --base-url https://api.example.com

  # JMeter JMX script
  testagent generate-perf -s swagger.json -f jmeter -o ./output/perf.jmx

  # Custom load profile
  testagent generate-perf -s swagger.json --virtual-users 50 --duration 120 --base-url https://api.example.com

  # Run generated scripts
  k6 run output/perf_test.js
  jmeter -n -t output/perf_test.jmx -l results.jtl
"""

GENERATE_GUI_EXAMPLES = """
EXAMPLES:
  # Generate Playwright test from requirements + target URL
  testagent generate-gui -r requirements.md --url https://example.com

  # With Swagger context for API-aware GUI tests
  testagent generate-gui -r requirements.md -s swagger.json --url https://app.example.com

  # Save to custom path
  testagent generate-gui -r requirements.md --url https://example.com -o tests/test_login.py

  # Run the generated script
  pytest output/gui_test.py --browser chromium
"""

CHAT_EXAMPLES = """
EXAMPLES:
  # Start interactive chat with requirements + swagger context
  testagent chat -r requirements.md -s swagger.json

  # Chat with just requirements
  testagent chat -r requirements.md

  # Chat with a specific session ID (resume previous session)
  testagent chat -r requirements.md --session my-session-1

  # Chat with a target URL so in-chat GUI test generation has a real app to test
  testagent chat -r requirements.md --url https://app.example.com

  # In the chat, type natural language:
  #   > Generate test cases for the user registration module
  #   > Add more boundary test cases for the password field
  #   > Generate a GUI test script for the login page
  #   > Validate the current test cases
  #   > Refine the test cases to be more concise
  #   > exit
"""

SERVE_EXAMPLES = """
EXAMPLES:
  # Start the web GUI on the default port (8000)
  testagent serve

  # Custom host/port
  testagent serve --host 0.0.0.0 --port 8080

  # Restrict iframe embedding to specific origins (default: allow all)
  WEB_FRAME_ANCESTORS="https://app.example.com https://portal.example.com" testagent serve

  # Embed in another platform via iframe
  #   <iframe src="http://localhost:8000/" width="100%" height="800"></iframe>

  # Install the web extra first if not already installed:
  #   pip install -e ".[web]"
"""
