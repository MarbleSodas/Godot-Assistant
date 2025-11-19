extends EditorDebuggerPlugin

signal log_message(entry: Dictionary)

var _capture_enabled: bool = false
var _message_buffer: Array[Dictionary] = []
var _max_buffer_size: int = 500

func _has_capture(message: String) -> bool:
	return _capture_enabled

func _capture_message(message: String, data: Array = []):
	if _capture_enabled:
		var entry = {
			"message": message,
			"data": data,
			"timestamp": Time.get_unix_time_from_system()
		}
		_message_buffer.append(entry)
		if _message_buffer.size() > _max_buffer_size:
			_message_buffer.pop_front()
		log_message.emit(entry)

func enable_capture():
	_capture_enabled = true

func disable_capture():
	_capture_enabled = false

func clear():
	_message_buffer.clear()

func get_buffer(limit: int = 200) -> Array[Dictionary]:
	if limit >= _message_buffer.size():
		return _message_buffer.duplicate()
	return _message_buffer.slice(_message_buffer.size() - limit)