# Rain Predictor Addon - Agent Guidelines

## Build/Lint/Test Commands

* `install`: Install Python dependencies (`pip3 install -r requirements.txt`)
* `test`: Run syntax checks (`python3 -m py_compile rain_predictor.py web_ui.py`)
* `run-web`: Start web UI only (`python3 web_ui.py`)
* `run-local`: Start both services locally (`./run_local.sh`)
* `docker-build`: Build Docker container (`docker build -t rain-predictor .`)
* `docker-run`: Run in Docker container (`docker run -p 8099:8099 rain-predictor`)

## Code Style Guidelines

### Python Standards
* **Imports**: Standard library first, then third-party, then local imports
* **Naming**: snake_case for variables/functions, PascalCase for classes
* **Docstrings**: Use triple quotes for all functions and classes
* **Error Handling**: Use try/except blocks, log errors appropriately
* **Types**: Use type hints for function parameters and return values

### JavaScript (Dashboard Card)
* **ES6+**: Use modern JavaScript features
* **Naming**: camelCase for variables and functions
* **Comments**: JSDoc style for functions

### YAML Configuration
* **Indentation**: 2 spaces
* **Comments**: Use # for comments
* **Structure**: Logical grouping of related settings

## Development Environment

### Local Development
1. Install dependencies: `pip3 install -r requirements.txt`
2. Create test config in `/tmp/test_data/options.json`
3. Run web UI: `python3 web_ui.py`
4. Access at http://localhost:8099

### Docker Development
1. Build: `docker build -t rain-predictor .`
2. Run: `docker run -p 8099:8099 -v /tmp/test_data:/data rain-predictor`

## Testing Strategy

### Unit Tests
- Test individual functions in `rain_predictor.py`
- Mock API calls and file operations
- Test edge cases for coordinate calculations

### Integration Tests
- Test full prediction cycle with sample radar data
- Test web UI API endpoints
- Test Home Assistant service calls

### Manual Testing
- Test with real RainViewer API
- Verify dashboard card integration
- Test configuration persistence

## Deployment

### Home Assistant Addon
- Repository: `https://github.com/AlliTec/rain-predictor-addon`
- Install via Add-on Store or local files
- Requires input_number entities in configuration.yaml

### Standalone Docker
- Build from Dockerfile
- Mount config volume at `/data`
- Expose port 8099 for web UI

## Key Configuration Files

* `config.yaml`: Home Assistant addon configuration
* `requirements.txt`: Python dependencies
* `Dockerfile`: Container build instructions
* `run.sh`: Production startup script
* `run_local.sh`: Local development startup script

## API Endpoints

### Web UI
- `GET /`: Main configuration interface
- `GET /api/data`: Current prediction data
- `POST /api/set_location`: Update location coordinates
- `POST /api/update_config`: Save configuration changes
- `POST /api/manual_selection`: Manual rain motion analysis

### Home Assistant Integration
- Updates input_number entities automatically
- Requires supervisor token for API access
- Uses `/data/options.json` for persistent configuration

## Troubleshooting

### Common Issues
1. **Missing dependencies**: Run `pip3 install -r requirements.txt`
2. **bashio not found**: Use `run_local.sh` instead of `run.sh`
3. **API connection errors**: Check internet connectivity and RainViewer API status
4. **Configuration not saving**: Verify `/data` directory permissions

### Debug Mode
- Set log level to DEBUG in configuration
- Enable save_images to capture radar tiles
- Check logs for detailed error messages
- Debug images saved to `/share/rain_predictor_debug/`

## Contributing

1. Follow code style guidelines
2. Add tests for new features
3. Update documentation
4. Test in both local and HA environments
5. Submit pull request with clear description