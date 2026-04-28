"""
dashboard.py — Tiny Flask web app for viewing and editing attendance.

Run:
    python dashboard.py

Then open http://localhost:5000 in your browser (any device on the same WiFi
can also use http://YOUR_LAPTOP_IP:5000 ).
"""

from datetime import date, datetime
from flask import Flask, render_template_string, request, redirect, url_for

import db


app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Attendance — {{ on_date }}</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif;
         max-width: 900px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1 { margin-bottom: 0; }
  .sub { color: #666; margin-top: 0.2em; }
  .controls { margin: 1.5em 0; display: flex; gap: 1em; align-items: center;
              flex-wrap: wrap; }
  .controls input[type=date] { padding: 0.4em; }
  .controls button, .controls a {
    padding: 0.5em 1em; border: 1px solid #444; background: #fff;
    cursor: pointer; border-radius: 4px; text-decoration: none; color: #222;
    font-size: 0.95em;
  }
  .controls button:hover, .controls a:hover { background: #f0f0f0; }
  .summary { padding: 0.8em 1em; background: #f4f4f4; border-radius: 6px;
             margin-bottom: 1em; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 0.6em 0.5em; text-align: left;
           border-bottom: 1px solid #eee; }
  th { background: #fafafa; font-size: 0.9em; color: #555; }
  tr.class-header td { background: #eef; font-weight: bold; padding-top: 1em; }
  .status-present { color: #1a7f1a; font-weight: 600; }
  .status-absent  { color: #b22222; font-weight: 600; }
  .status-not_marked { color: #888; font-style: italic; }
  .edit { font-size: 0.85em; color: #999; margin-left: 0.5em; }
  form.inline { display: inline; }
  form.inline button {
    padding: 0.3em 0.6em; font-size: 0.85em; cursor: pointer;
    border: 1px solid #bbb; background: #fff; border-radius: 3px;
  }
  .nav { margin-top: 2em; padding-top: 1em; border-top: 1px solid #eee;
         font-size: 0.9em; color: #777; }
</style>
</head>
<body>
  <h1>Attendance Register</h1>
  <p class="sub">{{ on_date }}{% if is_today %} (today){% endif %}</p>

  <div class="controls">
    <form method="get" action="/">
      <input type="date" name="date" value="{{ on_date }}">
      <button type="submit">Go</button>
    </form>
    {% if is_today %}
    <form method="post" action="/flag-absences" class="inline">
      <button type="submit"
        onclick="return confirm('Mark all unmarked students as absent for today?')">
        Flag remaining as absent
      </button>
    </form>
    {% endif %}
  </div>

  <div class="summary">
    <strong>Total:</strong> {{ rows|length }} &nbsp;|&nbsp;
    <strong style="color:#1a7f1a">Present:</strong> {{ counts.present }} &nbsp;|&nbsp;
    <strong style="color:#b22222">Absent:</strong> {{ counts.absent }} &nbsp;|&nbsp;
    <strong style="color:#888">Not marked:</strong> {{ counts.not_marked }}
  </div>

  <table>
    <thead>
      <tr><th>Name</th><th>Status</th><th>Arrival</th><th>Actions</th></tr>
    </thead>
    <tbody>
    {% set ns = namespace(current_class=None) %}
    {% for r in rows %}
      {% if r.class_name != ns.current_class %}
        {% set ns.current_class = r.class_name %}
        <tr class="class-header"><td colspan="4">{{ r.class_name }}</td></tr>
      {% endif %}
      <tr>
        <td>{{ r.name }}</td>
        <td>
          <span class="status-{{ r.status }}">
            {% if r.status == 'present' %}✓ Present
            {% elif r.status == 'absent' %}✗ Absent
            {% else %}— Not marked{% endif %}
          </span>
          {% if r.manually_edited %}<span class="edit">(edited)</span>{% endif %}
        </td>
        <td>{{ r.arrival_time or '—' }}</td>
        <td>
          <form method="post" action="/set-status" class="inline">
            <input type="hidden" name="student_id" value="{{ r.id }}">
            <input type="hidden" name="date" value="{{ on_date }}">
            <input type="hidden" name="status" value="present">
            <button type="submit" {% if r.status == 'present' %}disabled{% endif %}>
              Mark present
            </button>
          </form>
          <form method="post" action="/set-status" class="inline">
            <input type="hidden" name="student_id" value="{{ r.id }}">
            <input type="hidden" name="date" value="{{ on_date }}">
            <input type="hidden" name="status" value="absent">
            <button type="submit" {% if r.status == 'absent' %}disabled{% endif %}>
              Mark absent
            </button>
          </form>
        </td>
      </tr>
    {% endfor %}
    {% if not rows %}
      <tr><td colspan="4" style="padding:2em;text-align:center;color:#888">
        No students enrolled yet. Run <code>python enroll.py</code> to add some.
      </td></tr>
    {% endif %}
    </tbody>
  </table>

  <p class="nav">Tip: open this page on your phone using your laptop's IP address
  on the same WiFi.</p>
</body>
</html>
"""


@app.route("/")
def index():
    on_date = request.args.get("date") or date.today().isoformat()
    rows = db.get_attendance_for_date(on_date)
    counts = {"present": 0, "absent": 0, "not_marked": 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return render_template_string(
        PAGE,
        on_date=on_date,
        is_today=(on_date == date.today().isoformat()),
        rows=rows,
        counts=counts,
    )


@app.route("/set-status", methods=["POST"])
def set_status():
    student_id = int(request.form["student_id"])
    on_date = request.form.get("date") or date.today().isoformat()
    status = request.form["status"]
    db.set_attendance_status(student_id, status, on_date)
    return redirect(url_for("index", date=on_date))


@app.route("/flag-absences", methods=["POST"])
def flag_absences():
    db.flag_absences()
    return redirect(url_for("index"))


if __name__ == "__main__":
    db.init_db()
    # host=0.0.0.0 so other devices on the WiFi can reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
