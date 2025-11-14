from typing import Tuple, Dict, Any, List

def _get_value(obj: Dict[str, Any], path: str):
    parts = path.split('.')
    cur: Any = obj
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = None
    return cur

def _contains(value: Any, target: Any) -> bool:
    if isinstance(value, list):
        return str(target).lower() in {str(x).lower() for x in value}
    if isinstance(value, str):
        return str(target).lower() in value.lower()
    return False

def apply_custom_rules(profile: Dict[str, Any], rules: List[Dict[str, Any]]) -> Tuple[bool, str]:
    for r in rules or []:
        field = str(r.get('field', '')).strip()
        op = str(r.get('operator', '')).strip()
        val = r.get('value')
        v = _get_value(profile, field)
        if op == 'in':
            if isinstance(val, list):
                if isinstance(v, list):
                    if not any(_contains(val, x) for x in v):
                        return False, f'{field} not in {val}'
                else:
                    if str(v) not in {str(x) for x in val}:
                        return False, f'{field} not in {val}'
            else:
                if isinstance(v, list):
                    if not _contains(v, val):
                        return False, f'{field} not contains {val}'
                else:
                    if str(v) != str(val):
                        return False, f'{field} != {val}'
        elif op == 'not_in':
            if isinstance(val, list):
                if isinstance(v, list):
                    if any(_contains(val, x) for x in v):
                        return False, f'{field} has disallowed {val}'
                else:
                    if str(v) in {str(x) for x in val}:
                        return False, f'{field} disallowed {val}'
            else:
                if isinstance(v, list):
                    if _contains(v, val):
                        return False, f'{field} contains {val}'
                else:
                    if str(v) == str(val):
                        return False, f'{field} equals {val}'
        elif op == 'gt':
            try:
                if float(v or 0) <= float(val):
                    return False, f'{field} <= {val}'
            except Exception:
                return False, f'{field} invalid'
        elif op == 'lt':
            try:
                if float(v or 0) >= float(val):
                    return False, f'{field} >= {val}'
            except Exception:
                return False, f'{field} invalid'
        elif op == 'eq':
            if str(v) != str(val):
                return False, f'{field} != {val}'
        elif op == 'contains':
            if not _contains(v, val):
                return False, f'{field} not contains {val}'
    return True, 'ok'